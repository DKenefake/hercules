const EXPIRY_SLOTS: usize = 32;

struct BitSet {
    words: Vec<u64>,
    occupied: u64,
}

impl BitSet {
    fn new(words: usize) -> Self {
        Self {
            words: vec![0; words],
            occupied: 0,
        }
    }

    fn insert(&mut self, vertex: usize) {
        let word = vertex / 64;
        self.words[word] |= 1 << (vertex % 64);
        self.occupied |= 1 << word;
    }

    fn remove(&mut self, vertex: usize) {
        let word = vertex / 64;
        self.words[word] &= !(1 << (vertex % 64));
        if self.words[word] == 0 {
            self.occupied &= !(1 << word);
        }
    }

    fn first_at_or_after_cyclic(&self, offset: usize) -> usize {
        let word = offset / 64;
        let tail = self.words[word] & (u64::MAX << (offset % 64));
        if tail != 0 {
            return word * 64 + tail.trailing_zeros() as usize;
        }
        let later = (self.occupied >> word >> 1) << 1 << word;
        let next = (if later != 0 { later } else { self.occupied }).trailing_zeros() as usize;
        next * 64 + self.words[next].trailing_zeros() as usize
    }
}

struct GainSets {
    bins: Vec<BitSet>,
    occupied: u64,
    radius: i32,
}

impl GainSets {
    fn new(n: usize, radius: i32) -> Self {
        let words = n.div_ceil(64);
        assert!(words <= 64 && (0..32).contains(&radius));
        Self {
            bins: (0..=2 * radius).map(|_| BitSet::new(words)).collect(),
            occupied: 0,
            radius,
        }
    }

    fn bin(&self, delta: f64) -> usize {
        (self.radius + (delta * 0.5) as i32) as usize
    }

    fn insert(&mut self, i: usize, delta: f64) {
        let bin = self.bin(delta);
        self.bins[bin].insert(i);
        self.occupied |= 1 << bin;
    }

    fn remove(&mut self, i: usize, delta: f64) {
        let bin = self.bin(delta);
        self.bins[bin].remove(i);
        if self.bins[bin].occupied == 0 {
            self.occupied &= !(1 << bin);
        }
    }

    fn best(&self) -> Option<(f64, &BitSet)> {
        if self.occupied == 0 {
            return None;
        }
        let bin = self.occupied.trailing_zeros() as usize;
        Some((2.0 * f64::from(bin as i32 - self.radius), &self.bins[bin]))
    }
}

pub(super) fn flip_delta(x: u8, gradient: f64) -> f64 {
    if x == 0 {
        gradient
    } else {
        -gradient
    }
}

pub(super) struct GainBuckets {
    all: GainSets,
    eligible: GainSets,
    expiry: [Vec<usize>; EXPIRY_SLOTS],
}

impl GainBuckets {
    pub(super) fn new(x: &[u8], gradient: &[f64], radius: i32) -> Self {
        let mut all = GainSets::new(x.len(), radius);
        let mut eligible = GainSets::new(x.len(), radius);
        for (i, (&bit, &g)) in x.iter().zip(gradient).enumerate() {
            let delta = flip_delta(bit, g);
            all.insert(i, delta);
            eligible.insert(i, delta);
        }
        Self {
            all,
            eligible,
            expiry: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub(super) fn release(
        &mut self,
        step: usize,
        tabu_until: &[usize],
        x: &[u8],
        gradient: &[f64],
    ) {
        for i in self.expiry[step % EXPIRY_SLOTS].drain(..) {
            if tabu_until[i] == step {
                self.eligible.insert(i, flip_delta(x[i], gradient[i]));
            }
        }
    }

    pub(super) fn select(&self, offset: usize, value: f64, best: f64) -> Option<(usize, f64)> {
        // If the best gain passes aspiration, tabu status cannot exclude it.
        let all = self.all.best();
        let (delta, vertices) = match all {
            Some((delta, _)) if value + delta < best => all,
            _ => self.eligible.best(),
        }?;
        Some((vertices.first_at_or_after_cyclic(offset), delta))
    }

    pub(super) fn update(&mut self, i: usize, old: f64, new: f64, eligible: bool) {
        if old != new {
            self.all.remove(i, old);
            self.all.insert(i, new);
            if eligible {
                self.eligible.remove(i, old);
                self.eligible.insert(i, new);
            }
        }
    }

    pub(super) fn forbid(&mut self, i: usize, old: f64, eligible: bool, expires: usize) {
        self.update(i, old, -old, false);
        if eligible {
            self.eligible.remove(i, old);
        }
        self.expiry[expires % EXPIRY_SLOTS].push(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_ties_cross_word_boundaries_and_wrap() {
        let mut set = BitSet::new(64);
        for i in [3, 66, 130, 4094] {
            set.insert(i);
        }
        for (offset, expected) in [(0, 3), (4, 66), (66, 66), (67, 130), (4095, 3)] {
            assert_eq!(set.first_at_or_after_cyclic(offset), expected);
        }
    }

    #[test]
    fn aspiration_and_stale_expiry() {
        let (x, g) = ([0, 0], [-4.0, 0.0]);
        let mut buckets = GainBuckets::new(&[1, 0], &g, 2);
        buckets.forbid(0, 4.0, true, 9);
        assert_eq!(buckets.select(0, 10.0, 6.0), Some((1, 0.0)));
        assert_eq!(buckets.select(0, 10.0, 7.0), Some((0, -4.0)));
        buckets.forbid(0, -4.0, false, 33);
        buckets.release(9, &[33, 0], &[1, 0], &g);
        buckets.update(1, 0.0, -2.0, true);
        buckets.update(0, 4.0, -4.0, false);
        assert_eq!(buckets.select(0, 10.0, 6.0), Some((1, -2.0)));
        buckets.release(33, &[33, 0], &x, &g);
        assert_eq!(buckets.select(0, 10.0, 6.0), Some((0, -4.0)));
    }
}
