//! Stateful look-ahead PCM limiter corresponding to libPCMutils' `TDLimiter`.

/// FDK's decoder-side limiter defaults.
pub const DEFAULT_ATTACK_MS: u32 = 15;
pub const DEFAULT_RELEASE_MS: u32 = 50;

/// Floating-point port of the running-maximum time-domain limiter used by
/// libAACdec. Samples are represented internally in the normalized PCM range.
#[derive(Debug, Clone)]
pub struct TimeDomainLimiter {
    max_attack_ms: u32,
    attack_ms: u32,
    release_ms: u32,
    sample_rate: u32,
    channels: usize,
    attack_samples: usize,
    attack_const: f64,
    release_const: f64,
    maximum: f64,
    maximum_buffer: Vec<f64>,
    maximum_index: usize,
    delay_buffer: Vec<f64>,
    delay_index: usize,
    correction: f64,
    smooth_state: f64,
}

impl Default for TimeDomainLimiter {
    fn default() -> Self {
        Self::new(DEFAULT_ATTACK_MS, DEFAULT_RELEASE_MS)
    }
}

impl TimeDomainLimiter {
    pub fn new(max_attack_ms: u32, release_ms: u32) -> Self {
        let mut limiter = Self {
            max_attack_ms,
            attack_ms: max_attack_ms,
            release_ms,
            sample_rate: 0,
            channels: 0,
            attack_samples: 0,
            attack_const: 0.0,
            release_const: 0.0,
            maximum: 0.0,
            maximum_buffer: Vec::new(),
            maximum_index: 0,
            delay_buffer: Vec::new(),
            delay_index: 0,
            correction: 1.0,
            smooth_state: 1.0,
        };
        limiter.reset();
        limiter
    }

    pub fn attack_ms(&self) -> u32 {
        self.attack_ms
    }

    pub fn release_ms(&self) -> u32 {
        self.release_ms
    }

    pub fn set_attack_ms(&mut self, attack_ms: u32) -> bool {
        if attack_ms == 0 || attack_ms > self.max_attack_ms {
            return false;
        }
        if self.attack_ms != attack_ms {
            self.attack_ms = attack_ms;
            self.reconfigure(self.channels, self.sample_rate);
        }
        true
    }

    pub fn set_release_ms(&mut self, release_ms: u32) -> bool {
        if release_ms == 0 {
            return false;
        }
        if self.release_ms != release_ms {
            self.release_ms = release_ms;
            self.reconfigure(self.channels, self.sample_rate);
        }
        true
    }

    pub fn delay_samples(&self, sample_rate: u32) -> usize {
        (self.attack_ms as usize).saturating_mul(sample_rate as usize) / 1000
    }

    pub fn reset(&mut self) {
        self.maximum = 0.0;
        self.maximum_buffer.fill(0.0);
        self.maximum_index = 0;
        self.delay_buffer.fill(0.0);
        self.delay_index = 0;
        self.correction = 1.0;
        self.smooth_state = 1.0;
    }

    pub fn process_f32(&mut self, samples: &mut [f32], channels: usize, sample_rate: u32) {
        let mut normalized = samples
            .iter()
            .map(|&sample| sample as f64)
            .collect::<Vec<_>>();
        self.process_normalized(&mut normalized, channels, sample_rate);
        for (sample, limited) in samples.iter_mut().zip(normalized) {
            *sample = limited as f32;
        }
    }

    pub fn process_i16(&mut self, samples: &mut [i16], channels: usize, sample_rate: u32) {
        let mut normalized = samples
            .iter()
            .map(|&sample| sample as f64 / 32768.0)
            .collect::<Vec<_>>();
        self.process_normalized(&mut normalized, channels, sample_rate);
        for (sample, limited) in samples.iter_mut().zip(normalized) {
            *sample = (limited * 32768.0)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
    }

    fn reconfigure(&mut self, channels: usize, sample_rate: u32) {
        self.channels = channels;
        self.sample_rate = sample_rate;
        self.attack_samples = self.delay_samples(sample_rate);
        let release_samples =
            (self.release_ms as usize).saturating_mul(sample_rate as usize) / 1000;
        self.attack_const = 0.1f64.powf(1.0 / (self.attack_samples + 1) as f64);
        self.release_const = 0.1f64.powf(1.0 / (release_samples + 1) as f64);
        self.maximum_buffer = vec![0.0; self.attack_samples + 1];
        self.delay_buffer = vec![0.0; self.attack_samples.saturating_mul(channels)];
        self.reset();
    }

    fn process_normalized(&mut self, samples: &mut [f64], channels: usize, sample_rate: u32) {
        if channels == 0 || samples.is_empty() || !samples.len().is_multiple_of(channels) {
            return;
        }
        if self.channels != channels || self.sample_rate != sample_rate {
            self.reconfigure(channels, sample_rate);
        }
        if self.attack_samples == 0 {
            for sample in samples {
                *sample = sample.clamp(-1.0, 1.0);
            }
            return;
        }

        for frame in samples.chunks_exact_mut(channels) {
            let peak = frame
                .iter()
                .fold(1.0f64, |peak, sample| peak.max(sample.abs()));
            let old = self.maximum_buffer[self.maximum_index];
            self.maximum_buffer[self.maximum_index] = peak;
            if peak >= self.maximum {
                self.maximum = peak;
            } else if old >= self.maximum {
                self.maximum = self.maximum_buffer.iter().copied().fold(1.0f64, f64::max);
            }
            self.maximum_index = (self.maximum_index + 1) % self.maximum_buffer.len();

            let target = if self.maximum > 1.0 {
                1.0 / self.maximum
            } else {
                1.0
            };
            if target < self.smooth_state {
                self.correction = self
                    .correction
                    .min((target - 0.1 * self.smooth_state) / 0.9);
            } else {
                self.correction = target;
            }
            if self.correction < self.smooth_state {
                self.smooth_state =
                    self.attack_const * (self.smooth_state - self.correction) + self.correction;
                self.smooth_state = self.smooth_state.max(target);
            } else {
                self.smooth_state =
                    self.correction + self.release_const * (self.smooth_state - self.correction);
            }

            let delay_start = self.delay_index * channels;
            for (sample, delayed_sample) in frame
                .iter_mut()
                .zip(&mut self.delay_buffer[delay_start..delay_start.saturating_add(channels)])
            {
                let delayed = *delayed_sample;
                *delayed_sample = *sample;
                *sample = (delayed * self.smooth_state).clamp(-1.0, 1.0);
            }
            self.delay_index = (self.delay_index + 1) % self.attack_samples;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_transparent_signal_by_attack_time() {
        let mut limiter = TimeDomainLimiter::default();
        let mut samples = vec![0.0; 32];
        samples[0] = 0.5;
        limiter.process_f32(&mut samples, 1, 1000);
        assert_eq!(limiter.delay_samples(1000), 15);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[15], 0.5);
    }

    #[test]
    fn limits_linked_channels_and_validates_times() {
        let mut limiter = TimeDomainLimiter::default();
        assert!(!limiter.set_attack_ms(0));
        assert!(!limiter.set_attack_ms(16));
        assert!(limiter.set_attack_ms(1));
        assert!(!limiter.set_release_ms(0));
        assert!(limiter.set_release_ms(25));
        let mut samples = vec![2.0, 0.5, 0.0, 0.0];
        samples.extend(vec![0.0; 20]);
        limiter.process_f32(&mut samples, 2, 1000);
        assert!(samples.iter().all(|sample| sample.abs() <= 1.0));
    }
}
