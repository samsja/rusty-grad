use std::ops;

use ndarray::{Array, IxDyn, NdFloat};

use crate::variable::GradFn;
use crate::variable::VariableRef;

pub struct Add {}

impl<T> GradFn<T> for Add
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x + y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        _left_ref: &'a VariableRef<T>,
        _right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        println!("{}", grad);
        [grad.clone(), grad.clone()]
    }
}

pub struct Sub {}

impl<T> GradFn<T> for Sub
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x - y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        _left_ref: &'a VariableRef<T>,
        _right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        [grad.clone(), -grad.clone()]
    }
}

pub struct Mul {}

impl<T> GradFn<T> for Mul
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x * y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let left_var = left_ref.borrow();
        let right_var = right_ref.borrow();

        [right_var.data.clone() * grad, left_var.data.clone() * grad]
    }
}

pub struct Div {}

impl<T> GradFn<T> for Div
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn> {
        x / y
    }

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2] {
        let left_data = left_ref.borrow().data.clone();
        let right_data = right_ref.borrow().data.clone();

        [
            grad / right_data.clone(),
            -(grad * left_data) / (right_data.mapv(|a| a.powi(2))),
        ]
    }
}

#[macro_export]
macro_rules! impl_binary_op {
    ($trt:ident,  $mth:ident ) => {
        impl<'a, 'b, T> ops::$trt<&'b VariableRef<T>> for &'a VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: &'b VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, 'b, T> ops::$trt<&'b mut VariableRef<T>> for &'a VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: &'b mut VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, 'b, T> ops::$trt<&'b VariableRef<T>> for &'a mut VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: &'b VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, 'b, T> ops::$trt<&'b mut VariableRef<T>> for &'a mut VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: &'b mut VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, T> ops::$trt<&'a VariableRef<T>> for VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: &'a VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(&self, rhs, Box::new($trt {}))
            }
        }

        impl<'a, T> ops::$trt<VariableRef<T>> for &'a VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(self, &rhs, Box::new($trt {}))
            }
        }

        impl<T> ops::$trt<VariableRef<T>> for VariableRef<T>
        where
            T: NdFloat,
        {
            type Output = VariableRef<T>;

            fn $mth(self, rhs: VariableRef<T>) -> VariableRef<T> {
                let grad_fn = $trt {};
                grad_fn.subscribe(&self, &rhs, Box::new($trt {}))
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
        let ref x = Variable::new(array!([2.0]).into_dyn());
        let ref y = Variable::new(array!([3.0]).into_dyn());

        let mut z = x + y;

        z.backward();

        assert_eq!(x.borrow().get_grad_f(), array!([1.0]).into_dyn());
        assert_eq!(y.borrow().get_grad_f(), array!([1.0]).into_dyn());
    }

    #[test]
    fn sub_check_backward() {
        let ref x = Variable::new(array!([2.0]).into_dyn());
        let ref y = Variable::new(array!([3.0]).into_dyn());

        let mut z = x - y;

        z.backward();

        assert_eq!(x.borrow().get_grad_f(), array!([1.0]).into_dyn());
        assert_eq!(y.borrow().get_grad_f(), array!([-1.0]).into_dyn());
    }

    #[test]
    fn mul_check_backward() {
        let ref x = Variable::new(array!([2.0]).into_dyn());
        let ref y = Variable::new(array!([3.0]).into_dyn());

        let mut z = x * y;

        z.backward();

        assert_eq!(x.borrow().get_grad_f(), array!([3.0]).into_dyn());
        assert_eq!(y.borrow().get_grad_f(), array!([2.0]).into_dyn());
    }

    #[test]
    fn div_check_backward() {
        let ref x = Variable::new(array!([2.0]).into_dyn());
        let ref y = Variable::new(array!([3.0]).into_dyn());

        let mut z = x / y;

        z.backward();

        assert_eq!(x.borrow().get_grad_f(), array!([1.0 / 3.0]).into_dyn());
        assert_eq!(y.borrow().get_grad_f(), array!([-2.0 / 9.0]).into_dyn());
    }
}
