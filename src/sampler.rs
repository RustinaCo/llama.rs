//! Token sampling: temperature, top-k, top-p (nucleus) and repetition penalty.

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
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
        let mut probs: Vec<f32> = logits.to_vec();

        // repetition penalty
        if params.repetition_penalty != 1.0 {
            for &t in recent {
                let i = t as usize;
                if i < n {
                    let p = params.repetition_penalty;
                    if probs[i] > 0.0 {
                        probs[i] /= p;
                    } else {
                        probs[i] *= p;
                    }
                }
            }
        }

        if params.temperature <= 0.0 {
            // greedy
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
