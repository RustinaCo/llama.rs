//! Token sampling: temperature, top-k, top-p (nucleus) and repetition penalty.

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// How many trailing tokens the repetition penalty looks at. Matches the
    /// common `repeat-last-n` default; 0 disables the window (penalize all).
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0x1234_5678_9abc_def0,
        }
    }
}

pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler {
            state: seed | 1,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        // uniform in [0, 1)
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams, recent: &[u32]) -> u32 {
        let n = logits.len();
        if n == 0 {
            return 0;
        }
        // Fast zero-allocation path for greedy without repetition penalty
        if params.temperature <= 0.0 && params.repetition_penalty == 1.0 {
            let mut best = 0usize;
            for i in 1..n {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            return best as u32;
        }

        let mut probs: Vec<f32> = logits.to_vec();

        // repetition penalty
        //
        // Reference semantics (llama.cpp / OpenAI): look only at the last
        // `repeat_last_n` tokens and penalize each DISTINCT token at most
        // once. Penalizing the whole history — or multiplying the penalty by
        // the number of occurrences — drives the model off-distribution on
        // long prompts (it starts avoiding every token the prompt contains
        // and drifts into foreign scripts).
        if params.repetition_penalty != 1.0 {
            let p = params.repetition_penalty;
            let start = if params.repeat_last_n == 0 {
                0
            } else {
                recent.len().saturating_sub(params.repeat_last_n)
            };
            let mut seen: Vec<u32> = Vec::with_capacity(recent.len() - start);
            for &t in &recent[start..] {
                let i = t as usize;
                if i >= n || seen.contains(&t) {
                    continue;
                }
                seen.push(t);
                if probs[i] > 0.0 {
                    probs[i] /= p;
                } else {
                    probs[i] *= p;
                }
            }
        }

        if params.temperature <= 0.0 {
            // greedy with repetition penalty applied
            let mut best = 0usize;
            for i in 1..n {
                if probs[i] > probs[best] {
                    best = i;
                }
            }
            return best as u32;
        }

        let inv_t = 1.0 / params.temperature;
        for p in probs.iter_mut() {
            *p *= inv_t;
        }
        super::ops::softmax_inplace(&mut probs);

        // top-k
        let k = if params.top_k == 0 || params.top_k > n {
            n
        } else {
            params.top_k
        };
        let mut idx: Vec<usize> = (0..n).collect();
        if k < n {
            idx.select_nth_unstable_by(k - 1, |&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            idx.truncate(k);
        }
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        // top-p on the sorted list
        let mut cutoff = idx.len();
        if params.top_p > 0.0 && params.top_p < 1.0 {
            let mut cum = 0f32;
            for (pos, &i) in idx.iter().enumerate() {
                cum += probs[i];
                if cum >= params.top_p {
                    cutoff = pos + 1;
                    break;
                }
            }
        }
        idx.truncate(cutoff);

        let mut sum = 0f32;
        for &i in &idx {
            sum += probs[i];
        }
        if sum <= 0.0 {
            return idx[0] as u32;
        }
        let mut r = self.next_f32() * sum;
        for &i in &idx {
            r -= probs[i];
            if r <= 0.0 {
                return i as u32;
            }
        }
        idx[idx.len() - 1] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greedy_params(penalty: f32, last_n: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: penalty,
            repeat_last_n: last_n,
            seed: 1,
        }
    }

    #[test]
    fn repetition_penalty_applies_once_per_distinct_token() {
        // token 0 seen 4x; penalizing per-occurrence would give 10 / 2^4 = 0.625
        // and flip the argmax to token 1. One penalty keeps it at 5.0.
        let logits = vec![10.0f32, 1.0];
        let recent = vec![0u32, 0, 0, 0];
        let mut s = Sampler::new(1);
        assert_eq!(s.sample(&logits, &greedy_params(2.0, 64), &recent), 0);
    }

    #[test]
    fn repetition_penalty_respects_window() {
        let logits = vec![10.0f32, 1.0];
        let mut s = Sampler::new(1);
        // token 0 inside the 2-token window -> 10 / 20 = 0.5 -> token 1 wins
        assert_eq!(s.sample(&logits, &greedy_params(20.0, 2), &[7, 7, 0]), 1);
        // token 0 pushed out of the window -> untouched -> token 0 wins
        assert_eq!(s.sample(&logits, &greedy_params(20.0, 2), &[0, 7, 7]), 0);
    }

    #[test]
    fn repetition_penalty_disabled_is_identity() {
        let logits = vec![10.0f32, 1.0];
        let mut s = Sampler::new(1);
        assert_eq!(s.sample(&logits, &greedy_params(1.0, 64), &[0, 0, 0]), 0);
    }
}
