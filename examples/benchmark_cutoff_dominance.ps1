param(
    [string]$OutputDirectory = 'target/presolve-benchmark/cutoff-dominance',
    [int]$Seconds = 30,
    [int]$HardSeconds = 45
)
$ErrorActionPreference = 'Stop'
if ($Seconds -le 0 -or $HardSeconds -le $Seconds) { throw 'Require 0 < Seconds < HardSeconds' }
if (Test-Path (Join-Path $OutputDirectory 'results.json')) { throw 'Choose a new output directory' }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$results = [System.Collections.Generic.List[object]]::new()
$modes = @(
    @{name='baseline'; cutoff='false'; dominance='false'},
    @{name='cutoff'; cutoff='true'; dominance='false'},
    @{name='cutoff-dominance'; cutoff='true'; dominance='true'}
)
foreach ($instance in @('mk487a', 'mk487b', 'mk487c')) {
    foreach ($mode in $modes) {
        $log = Join-Path $OutputDirectory "$instance-$($mode.name).log"
        Write-Output "START $instance $($mode.name)"
        docker exec -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -w /work hercules-dev `
            timeout --signal=TERM --kill-after=5s "${HardSeconds}s" `
            /target/release/examples/profile_presolve --solve "test_data/$instance.qubo" `
            0 $Seconds 1 256 0.01 false false false 0 0.25 true true true true true LargestEdges `
            true false $mode.cutoff $mode.dominance 2>&1 | Tee-Object -FilePath $log | ForEach-Object {
                if ($_ -match '^(ROOT_REDUCTION_CONFIG |REDUCTION |NODE_REDUCTION |PROBING |CUTOFF |END_TO_END |VALIDATION )') { Write-Output $_ }
            }
        $code = $LASTEXITCODE
        $lines = Get-Content -LiteralPath $log
        $end = $lines | Where-Object { $_ -match '^END_TO_END ' } | Select-Object -Last 1
        $row = [ordered]@{ instance=$instance; mode=$mode.name; exit_code=$code; log=$log }
        if ($code -ne 0 -or -not $end) { throw "Run failed: $log ($code)" }
        foreach ($pair in [regex]::Matches($end, '(\w+)=(\S+)')) { $row[$pair.Groups[1].Value] = $pair.Groups[2].Value }
        foreach ($prefix in @('PROBING','REDUCTION','NODE_REDUCTION','CUTOFF')) {
            $row[$prefix.ToLower()] = $lines | Where-Object { $_ -match "^$prefix " } | Select-Object -Last 1
        }
        if (-not ($lines | Where-Object { $_ -match '^VALIDATION binary=true ' })) { throw "Invalid solution: $log" }
        $objective = [double]::Parse($row.objective, [cultureinfo]::InvariantCulture)
        $lower = [double]::Parse($row.lower_bound, [cultureinfo]::InvariantCulture)
        $expected = switch ($instance) { 'mk487a' { -1110926.0 } 'mk487b' { -3655475.0 } }
        if ($lower -gt $objective + 1e-5 -or ($null -ne $expected -and (
            $lower -gt $expected + 1e-5 -or $objective -lt $expected - 1e-5 -or
            ($row.status -eq 'Optimal' -and [Math]::Abs($objective - $expected) -gt 1e-5)
        ))) { throw "Incorrect bound or objective: $log" }
        $results.Add([pscustomobject]$row)
        $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'results.json')
    }
}
