#[derive(Debug)]
pub struct Arena<T> {
    nodes: Vec<Node<T>>,
}

#[derive(Debug)]
pub struct Node<T> {
    left_root: Option<NodeId>,
    right_root: Option<NodeId>,
    pub data: T,
}

#[derive(Debug)]
pub struct NodeId {
    index: usize,
}
impl<T> Arena<T> {
    pub fn new() -> Arena<T> {
        Self::default()
    }

    pub fn new_leaf(&mut self, data: T) -> NodeId {
        self.new_node(data, None, None)
    }

    pub fn new_node(
        &mut self,
        data: T,
        left_root: Option<NodeId>,
        right_root: Option<NodeId>,
    ) -> NodeId {
        let next_index = self.nodes.len();

        self.nodes.push(Node {
            left_root: left_root,
            right_root: right_root,
            data: data,
        });

        NodeId { index: next_index }
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self { nodes: Vec::new() }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn arena_test() {
        let arena = &mut Arena::<f32>::new();

        let n: f32 = 1.0;
        let a = arena.new_leaf(n);

        assert_eq!(arena.nodes[0].data, n)
    }
}
