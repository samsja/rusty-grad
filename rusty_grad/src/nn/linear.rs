use ndarray::{Array, Ix1, Ix2};

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Array<f32, Ix2>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Linear {
        Linear {
            in_features,
            out_features,
            weight: Linear::init_weight(in_features, out_features),
        }
    }

    fn init_weight(in_features: usize, out_features: usize) -> Array<f32, Ix2> {
        Array::<f32, Ix2>::zeros((in_features, out_features))
    }

    pub fn forward(&self, input: &Array<f32, Ix1>) -> Array<f32, Ix1> {
        input.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn() {
        let ref x = Array::<f32, _>::zeros(10);
        let shape = x.shape().clone();
        println!("shape {:#?}", shape);

        let layer1 = Linear::new(10, 2);

        let y = layer1.forward(x);
        println!("shape {:#?}", y.shape());

        assert_eq!(shape, y.shape());
    }
}
