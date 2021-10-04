use ndarray::{concatenate, stack, Array, Axis, Ix1, Ix2};

use crate::variable::{Variable, VariableRef};

pub fn make_moon(n_samples: usize) -> [Array<f32, Ix2>; 2] {
    let n_samples_in = n_samples / 2;
    let n_samples_out = n_samples - n_samples_in;

    let pi = std::f32::consts::PI;

    let out_circ_x = Array::linspace(0., pi, n_samples_out).mapv(|x| x.cos());
    let out_circ_y = Array::linspace(0., pi, n_samples_out).mapv(|y| y.sin());

    let in_circ_x = Array::linspace(0., pi, n_samples_in).mapv(|x| 1. - x.cos());
    let in_circ_y = Array::linspace(0., pi, n_samples_in).mapv(|x| 1. - x.sin() - 0.5);

    let out_circ = stack(Axis(0), &[out_circ_x.view(), out_circ_y.view()]).unwrap();
    let in_circ = stack(Axis(0), &[in_circ_x.view(), in_circ_y.view()]).unwrap();

    [out_circ, in_circ]
}

pub struct MakeMoonDataset {
    data: Array<f32, Ix2>,
    label: Array<f32, Ix1>,
}

impl MakeMoonDataset {
    pub fn new(n_samples: usize) -> MakeMoonDataset {
        let [out_circ, in_circ] = make_moon(n_samples);

        let data = concatenate(Axis(1), &[in_circ.view(), out_circ.view()]).unwrap();

        let label_out = Array::<f32, Ix1>::zeros(out_circ.shape()[1]);
        let label_in = Array::<f32, Ix1>::ones(in_circ.shape()[1]);

        let label = concatenate(Axis(0), &[label_in.view(), label_out.view()]).unwrap();

        MakeMoonDataset { data, label }
    }

    pub fn len(&self) -> usize {
        self.data.shape()[1]
    }

    pub fn get(&self, idx: usize) -> (VariableRef<f32>, f32) {
        let data = self.data.column(idx).to_shape((2, 1)).unwrap().mapv(|x| x);
        let data_var = Variable::new_no_retain_grad(data.into_dyn());
        (data_var, self.label[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_moon_test() {
        let n: usize = 100;
        let [out_circ, in_circ] = make_moon(2 * n);

        assert_eq!(out_circ.shape(), [2, 100]);
        assert_eq!(in_circ.shape(), [2, 100]);
    }

    #[test]
    fn make_moon_dataset_test() {
        let n: usize = 100;

        let dataset = MakeMoonDataset::new(n);

        let data = dataset.get(0);

        assert_eq!(data.0.borrow().data.shape(), [2, 1]);
        assert_eq!(data.1, 1.);
    }
}
