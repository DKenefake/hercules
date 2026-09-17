param(
    [string]$OutputDirectory = 'target/presolve-benchmark/branching-rules',
    [int]$Seconds = 30,
    [int]$HardSeconds = 45
)

$ErrorActionPreference = 'Stop'
if ($Seconds -le 0 -or $HardSeconds -le $Seconds) {
    throw 'Require 0 < Seconds < HardSeconds'
}
if (Test-Path (Join-Path $OutputDirectory 'results.json')) {
    throw 'Choose a new output directory rather than overwriting previous results'
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$rules = @(
    'LargestEdges', 'MostEdges', 'FirstNotFixed', 'MostViolated', 'Random',
    'WorstApproximation', 'WorstApproximation2', 'MostFixed', 'RoundRobin',
    'LargestDiag', 'MovingEdges', 'ConnectedComponents'
)
$results = [System.Collections.Generic.List[object]]::new()
$culture = [System.Globalization.CultureInfo]::InvariantCulture
foreach ($instance in @('mk487a', 'mk487b', 'mk487c')) {
    foreach ($rule in $rules) {
        $log = Join-Path $OutputDirectory "$instance-$rule.log"
        Write-Output "START instance=$instance rule=$rule limit=$Seconds"
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        docker exec -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -w /work hercules-dev `
            timeout --signal=TERM --kill-after=5s "${HardSeconds}s" `
            /target/release/examples/profile_presolve --solve "test_data/$instance.qubo" `
            0 $Seconds 1 256 0.01 false false false 0 0.25 true true true true true $rule `
            2>&1 | Tee-Object -FilePath $log | ForEach-Object {
                if ($_ -match '^(CONFIG |ROOT_REDUCTION_CONFIG |VALIDATION |END_TO_END )') {
                    Write-Output $_
                }
            }
        $code = $LASTEXITCODE
        $watch.Stop()
        $lines = Get-Content -LiteralPath $log
        $end = $lines | Where-Object { $_ -match '^END_TO_END ' } | Select-Object -Last 1
        $row = [ordered]@{
            instance = $instance; rule = $rule; exit_code = $code
            status = if ($code -in @(124, 137)) { 'HardTimeout' } else { 'Error' }
            wall_seconds = $watch.Elapsed.TotalSeconds
            seconds = $null; visited = $null; sdp_calls = $null; splits = $null
            objective = $null; lower_bound = $null; gap_percent = $null
            remaining = $null; validated = $false; log = $log
        }
        if ($code -eq 0 -and $end) {
            $fields = @{}
            foreach ($pair in [regex]::Matches($end, '(\w+)=(\S+)')) {
                $fields[$pair.Groups[1].Value] = $pair.Groups[2].Value
            }
            $row.status = $fields.status
            foreach ($key in @('seconds', 'objective', 'lower_bound')) {
                $row[$key] = [double]::Parse($fields[$key], $culture)
            }
            foreach ($key in @('visited', 'remaining')) {
                $row[$key] = [long]::Parse($fields[$key], $culture)
            }
            $row.gap_percent = 100 * ($row.objective - $row.lower_bound) / [Math]::Max(1, [Math]::Abs($row.objective))
            $row.validated = [bool]($lines | Where-Object { $_ -match '^VALIDATION binary=true ' })
            $probe = $lines | Where-Object { $_ -match '^PROBING ' } | Select-Object -Last 1
            if ($probe -match 'subproblem_calls: (\d+)') { $row.sdp_calls = [long]$Matches[1] }
            $decomp = $lines | Where-Object { $_ -match '^DECOMPOSITION ' } | Select-Object -Last 1
            if ($decomp -match 'splits: (\d+)') { $row.splits = [long]$Matches[1] }
            if (-not $row.validated -or $row.lower_bound -gt $row.objective + 1e-5) {
                $row.status = 'ValidationError'
            }
            $expected = switch ($instance) { 'mk487a' { -1110926.0 } 'mk487b' { -3655475.0 } }
            if ($null -ne $expected -and (
                $row.lower_bound -gt $expected + 1e-5 -or $row.objective -lt $expected - 1e-5 -or
                ($row.status -eq 'Optimal' -and [Math]::Abs($row.objective - $expected) -gt 1e-5)
            )) { $row.status = 'ValidationError' }
        }
        $results.Add([pscustomobject]$row)
        $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'results.json')
        Write-Output "RESULT instance=$instance rule=$rule status=$($row.status) seconds=$($row.seconds) visited=$($row.visited) sdp=$($row.sdp_calls) gap=$($row.gap_percent)"
    }
}
if ($results | Where-Object { $_.status -in @('Error', 'ValidationError') }) {
    throw 'Some runs failed; inspect results.json and individual logs'
}
