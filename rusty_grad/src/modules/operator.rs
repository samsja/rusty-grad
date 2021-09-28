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

pub struct Sub {}

impl<T, D> Module<T, D> for Sub
where
    T: NdFloat,
    D: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, y: &'b Array<T, D>) -> Array<T, D> {
        x - y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        _left_ref: &'a VariableRef<T, D>,
        _right_ref: &'a VariableRef<T, D>,
    ) -> [Array<T, D>; 2] {
        [grad.clone(), -grad.clone()]
    }
}

pub struct Mul {}

impl<T, D> Module<T, D> for Mul
where
    T: NdFloat,
    D: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, y: &'b Array<T, D>) -> Array<T, D> {
        x * y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        left_ref: &'a VariableRef<T, D>,
        right_ref: &'a VariableRef<T, D>,
    ) -> [Array<T, D>; 2] {
        let left_var = left_ref.borrow();
        let right_var = right_ref.borrow();

        [right_var.data.clone() * grad, left_var.data.clone() * grad]
    }
}

pub struct Div {}

impl<T, D> Module<T, D> for Div
where
    T: NdFloat,
    D: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, y: &'b Array<T, D>) -> Array<T, D> {
        x / y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        left_ref: &'a VariableRef<T, D>,
        right_ref: &'a VariableRef<T, D>,
    ) -> [Array<T, D>; 2] {
        let left_var = left_ref.borrow();
        let right_var = right_ref.borrow();

        [
            grad / right_var.data.clone(),
            (grad / left_var.data.clone()),
        ] // TODO should be minus
    }
}

#[macro_export]
macro_rules! impl_binary_op {
    ($trt:ident,  $mth:ident ) => {
        impl<'a, 'b, T, D> ops::$trt<&'b VariableRef<T, D>> for &'a VariableRef<T, D>
        where
            T: NdFloat,
            D: Dimension,
        {
            type Output = VariableRef<T, D>;

            fn $mth(self, rhs: &'b VariableRef<T, D>) -> VariableRef<T, D> {
                let module = $trt {};
                module.subscribe(self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, T, D> ops::$trt<&'a VariableRef<T, D>> for VariableRef<T, D>
        where
            T: NdFloat,
            D: Dimension,
        {
            type Output = VariableRef<T, D>;

            fn $mth(self, rhs: &'a VariableRef<T, D>) -> VariableRef<T, D> {
                let module = $trt {};
                module.subscribe(&self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, T, D> ops::$trt<VariableRef<T, D>> for &'a VariableRef<T, D>
        where
            T: NdFloat,
            D: Dimension,
        {
            type Output = VariableRef<T, D>;

            fn $mth(self, rhs: VariableRef<T, D>) -> VariableRef<T, D> {
                let module = $trt {};
                module.subscribe(self, &rhs, Box::new($trt {}))
            }
        }

        impl<T, D> ops::$trt<VariableRef<T, D>> for VariableRef<T, D>
        where
            T: NdFloat,
            D: Dimension,
        {
            type Output = VariableRef<T, D>;

            fn $mth(self, rhs: VariableRef<T, D>) -> VariableRef<T, D> {
                let module = $trt {};
                module.subscribe(&self, &rhs, Box::new($trt {}))
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);

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

    #[test]
    fn sub_check_backward() {
        let ref x = Variable::new(array!([2.0]));
        let ref y = Variable::new(array!([3.0]));

        let mut z = x - y;

        z.backward();

        assert_eq!(x.borrow().grad, array!([1.0]));
        assert_eq!(y.borrow().grad, array!([-1.0]));
    }

    #[test]
    fn mul_check_backward() {
        let ref x = Variable::new(array!([2.0]));
        let ref y = Variable::new(array!([3.0]));

        let mut z = x * y;

        z.backward();

        assert_eq!(x.borrow().grad, array!([3.0]));
        assert_eq!(y.borrow().grad, array!([2.0]));
    }

    #[test]
    fn div_check_backward() {
        let ref x = Variable::new(array!([2.0]));
        let ref y = Variable::new(array!([3.0]));

        let mut z = x * y;

        z.backward();

        assert_eq!(x.borrow().grad, array!([1.0 / 3.0]));
        assert_eq!(y.borrow().grad, array!([-2.0 / 9.0]));
    }
}