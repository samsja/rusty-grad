use crate::variable::VariableRef;

use ndarray::NdFloat;

pub trait Module<T: NdFloat> {
    fn params(&self) -> Vec<VariableRef<T>>;

    fn f(&mut self, input: &VariableRef<T>) -> VariableRef<T>;
}
