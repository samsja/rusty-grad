use std::fmt;
// *** NUM DATA *******

#[derive(Debug)]
pub struct NumData {
    data: f32,
}

impl NumData {
    fn new(data: f32) -> NumData {
        NumData { data }
    }

    fn zero() -> NumData {
        NumData { data: 0.0 }
    }
}

impl fmt::Display for NumData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

#[derive(Debug)]
pub struct Variable2Id {
    index: usize,
}

impl Variable2Id {
    pub fn new(index: usize) -> Variable2Id {
        Variable2Id { index }
    }
}

#[derive(Debug)]
pub struct Variable2 {
    pub data: NumData,
    pub grad: NumData,
    pub node_id: Variable2Id,
    left_root: Option<Variable2Id>,
    right_root: Option<Variable2Id>,

    graph: Option<Vec<Variable2Id>>,
}

// ********************** INIT **********************************
impl Variable2 {
    pub fn new_node(
        data: NumData,
        node_id: Variable2Id,
        left_root: Option<Variable2Id>,
        right_root: Option<Variable2Id>,
        graph: Option<Vec<Variable2Id>>,
    ) -> Variable2 {
        Variable2 {
            data,
            grad: NumData::zero(),
            node_id,
            left_root,
            right_root,
            graph,
        }
    }
}

impl Variable2 {
    pub fn new(data: NumData) -> Variable2 {
        let graph = Vec::new();

        Variable2::new_node(data, Variable2Id::new(0), None, None, Some(graph))
    }
}

impl fmt::Display for Variable2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Variable( {} grad : {})", self.data, self.grad)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn variable_2_init() {
        let n = 2.0;
        let var = Variable2::new(NumData::new(n));

        assert_eq!(var.data.data, NumData::new(n).data)
    }
}
