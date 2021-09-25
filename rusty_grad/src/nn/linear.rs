use ndarray::{Array, Ix1, Ix2};

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Array<f32, Ix2>,
    bias: Array<f32, Ix1>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Linear {
        Linear {
            in_features,
            out_features,
            weight: Linear::init_weight(in_features, out_features),
            bias: Linear::init_bias(out_features),
        }
    }

    fn init_weight(in_features: usize, out_features: usize) -> Array<f32, Ix2> {
        Array::<f32, Ix2>::zeros((out_features, in_features))
    }

    fn init_bias(out_features: usize) -> Array<f32, Ix1> {
        Array::<f32, Ix1>::zeros(out_features)
    }

    pub fn forward(&self, input: &Array<f32, Ix1>) -> Array<f32, Ix1> {
        let output = input.clone();
        self.weight.dot(&output) + self.bias.clone()
    }
}

pub struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    pub fn forward(&self, input: &Array<f32, Ix1>) -> Array<f32, Ix1> {
        let mut output = input.clone();

        for lay in self.layers.iter() {
            output = lay.forward(&output);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_forward() {
        let ref x = Array::<f32, _>::zeros(10);

        let layer = Linear::new(10, 2);

        let y = layer.forward(x);

        let out = Array::<f32, _>::zeros(2);
        let shape = out.shape();
        assert_eq!(shape, y.shape());
    }

    #[test]
    fn mlp_forward() {
        let layer1 = Linear::new(5, 10);
        let layer2 = Linear::new(10, 10);
        let layer3 = Linear::new(10, 2);

        let mlp = MLP {
            layers: vec![layer1, layer2, layer3],
        };

        let ref x = Array::<f32, Ix1>::zeros(5);

        let y = mlp.forward(x);

        let out = Array::<f32, _>::zeros(2);
        let shape = out.shape();
        assert_eq!(shape, y.shape());
    }
}
