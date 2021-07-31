// use crate::variable::Operator;
use crate::variable::Variable;
// use crate::variable::VariableRef;

// ************************ unit tests ******************************

#[derive(Debug)]
pub struct Linear {
    weight: Vec<Variable>,
    bias: Vec<Variable>,
    in_features: i32,
    out_features: i32,
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn init_test() {
        assert_eq!(true, true);
    }
}
