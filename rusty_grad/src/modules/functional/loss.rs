use ndarray::{Array, Ix1, IxDyn, NdFloat, Zip};
use num_traits::FromPrimitive;
use std::ops::Mul;

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct MSEloss {}

impl<T> Module<T> for MSEloss
where
    T: NdFloat + FromPrimitive + Mul<f32, Output = T>,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        let mut mse = x.clone();
        Zip::from(&mut mse).and(y).for_each(|a, &b| {
            *a = (*a - b) * (*a - b);
        });

        let mse = mse.mean().unwrap();
        Array::<T, Ix1>::ones(1).into_dyn() * mse
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let ref x = left_ref.borrow().data;
        let ref y = right_ref.borrow().data;

        let len = T::from_usize(x.len()).unwrap();
        let mut new_grad = ((y - x) / len) * 2.0;
        new_grad = new_grad * grad;
        [new_grad.clone(), new_grad]
    }
}

pub fn mse_loss<T: NdFloat + FromPrimitive + Mul<f32, Output = T>>(
    x: &VariableRef<T>,
    y: &VariableRef<T>,
) -> VariableRef<T> {
    let module = MSEloss {};
    module.subscribe(x, y, Box::new(MSEloss {}))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::variable::Variable;
    use ndarray::array;

    #[test]
    fn check_forward_vect() {
        let x = Variable::new(array!([1.0], [1.0]).into_dyn());
        let y = Variable::new(array!([3.0], [0.0]).into_dyn());

        let z = mse_loss(&x, &y);

        println!("{}", x.borrow().data.sum());
        assert_eq!(z.borrow().data, array!(2.5).into_dyn());
    }

    #[test]
    fn check_forward_mat() {
        let x = Variable::new(array!([1.0, 2.0], [3.0, 4.0]).into_dyn());
        let y = Variable::new(array!([0.0, -2.0], [3.0, 5.0]).into_dyn());

        let z = mse_loss(&x, &y);
        assert_eq!(z.borrow().data, array!(4.5).into_dyn());
    }

    #[test]
    fn mse_check_backward_mat() {
        let x = Variable::new(array!([1.0, 2.0], [3.0, 4.0]).into_dyn());
        let y = Variable::new(array!([0.0, -2.0], [3.0, 5.0]).into_dyn());

        let mut z = mse_loss(&x, &y);

        println!("{}", array!([1.0]) * 2.0);

        z.backward();
        assert_eq!(
            x.borrow().get_grad_f(),
            array!([-0.5, -2.0], [0.0, 0.5]).into_dyn()
        );
        assert_eq!(
            y.borrow().get_grad_f(),
            array!([-0.5, -2.0], [0.0, 0.5]).into_dyn()
        );
    }
}
