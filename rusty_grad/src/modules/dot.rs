use ndarray::{stack, ArrayView, Axis};
use ndarray::{Array, Ix2, IxDyn, NdFloat};
use ndarray::{Dimension, RemoveAxis, ShapeError};

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Dot {}

impl<T> Module<T> for Dot
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        let x = x.clone().into_dimensionality::<Ix2>().unwrap();
        let y = y.clone().into_dimensionality::<Ix2>().unwrap();

        x.dot(&y).into_dyn()
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        // Still in dev : need to implement the right_grad as well and to do for the full matrix product

        let ref x = left_ref.borrow().data;
        let ref y = right_ref.borrow().data;

        let x = x.clone().into_dimensionality::<Ix2>().unwrap();
        let y = y.clone().into_dimensionality::<Ix2>().unwrap();

        let grad_view = grad.view();
        let [grad_x, grad_y] =
            Dot::backward_ix2(&grad_view.into_dimensionality::<Ix2>().unwrap(), &x, &y);

        [grad_x.into_dyn(), grad_y.into_dyn()]
    }
}

impl Dot {
    fn backward_ix2<T: NdFloat>(
        grad: &ArrayView<T, Ix2>,
        x: &Array<T, Ix2>,
        y: &Array<T, Ix2>,
    ) -> [Array<T, Ix2>; 2] {
        let grad_x = y.sum_axis(Axis(1));

        let grad_x = repeat(Axis(0), &grad_x, x.shape()[1]).unwrap();

        let grad_y = x.sum_axis(Axis(0));
        let ax = if y.shape()[1] == 1 { 1 } else { 0 };

        let grad_y = repeat(Axis(ax), &grad_y, y.shape()[1]).unwrap();

        [grad * grad_x, grad * grad_y]
    }
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn dot(&mut self, other: &VariableRef<T>) -> VariableRef<T> {
        let module = Dot {};
        module.subscribe(self, other, Box::new(Dot {}))
    }
}

pub fn repeat<T, D>(ax: Axis, x: &Array<T, D>, n: usize) -> Result<Array<T, D::Larger>, ShapeError>
where
    T: NdFloat,
    D: Dimension,
    D::Larger: RemoveAxis,
{
    let l: Vec<ArrayView<T, D>> = (0..n).map(|_| x.view()).collect();
    stack(ax, &l)
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;
    use ndarray::array;

    #[test]
    fn check_method() {
        let mut x = Variable::new(array!([1.0, 1.0], [1.0, 1.0]).into_dyn());
        let y = Variable::new(array!([1.0], [1.0]).into_dyn());

        let res = array!([2.0], [2.0]).into_dyn();

        assert_eq!(x.dot(&y).borrow().data, res);
    }

    #[test]
    fn dot_check_backward_mat_vect() {
        let mut x = Variable::new(array!([1.0, 2.0], [3.0, 4.0]).into_dyn());
        let y = Variable::new(array!([1.0], [2.0]).into_dyn());

        let mut z = x.dot(&y);

        z.backward();
        assert_eq!(
            x.borrow().get_grad_f(),
            array!([1.0, 2.0], [1.0, 2.0]).into_dyn()
        );
        assert_eq!(y.borrow().get_grad_f(), array!([4.0], [6.0]).into_dyn());
    }

    #[test]
    fn dot_check_backward_mat_mat() {
        let mut x = Variable::new(array!([1.0, 2.0], [3.0, 4.0]).into_dyn());
        let y = Variable::new(array!([1.0, 1.0], [1.0, 1.0]).into_dyn());

        let mut z = x.dot(&y);

        z.backward();
        assert_eq!(
            x.borrow().get_grad_f(),
            array!([2.0, 2.0], [2.0, 2.0]).into_dyn()
        );
        assert_eq!(
            y.borrow().get_grad_f(),
            array!([4.0, 6.0], [4.0, 6.0]).into_dyn()
        );
    }
}
