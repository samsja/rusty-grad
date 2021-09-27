use std::ops;

use ndarray::{Array, Dimension, NdFloat};

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Add {}

impl<T, D> Module<T, D> for Add
where
    T: NdFloat,
    D: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, y: &'b Array<T, D>) -> Array<T, D> {
        x + y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        _left_ref: &'a VariableRef<T, D>,
        _right_ref: &'a VariableRef<T, D>,
    ) -> [Array<T, D>; 2] {
        [grad.clone(), grad.clone()]
    }
}

impl<'a, 'b, T, D> ops::Add<&'b VariableRef<T, D>> for &'a VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    type Output = VariableRef<T, D>;

    fn add(self, rhs: &'b VariableRef<T, D>) -> VariableRef<T, D> {
        let module = Add {};
        module.subscribe(self, rhs, Box::new(Add {}))
    }
}

impl<'a, T, D> ops::Add<&'a VariableRef<T, D>> for VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    type Output = VariableRef<T, D>;

    fn add(self, rhs: &'a VariableRef<T, D>) -> VariableRef<T, D> {
        let module = Add {};
        module.subscribe(&self, rhs, Box::new(Add {}))
    }
}

impl<'a, T, D> ops::Add<VariableRef<T, D>> for &'a VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    type Output = VariableRef<T, D>;

    fn add(self, rhs: VariableRef<T, D>) -> VariableRef<T, D> {
        let module = Add {};
        module.subscribe(self, &rhs, Box::new(Add {}))
    }
}

impl<T, D> ops::Add<VariableRef<T, D>> for VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    type Output = VariableRef<T, D>;

    fn add(self, rhs: VariableRef<T, D>) -> VariableRef<T, D> {
        let module = Add {};
        module.subscribe(&self, &rhs, Box::new(Add {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;
    use ndarray::array;

    #[test]
    fn add_check_backward() {
        let ref x = Variable::new(array!([2.0]));
        let ref y = Variable::new(array!([3.0]));

        let mut z = x + y;

        z.backward();

        assert_eq!(x.borrow().grad, array!([1.0]));
        assert_eq!(y.borrow().grad, array!([1.0]));
    }
}
