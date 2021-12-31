public class TreeTesting extends TreeUtilities{
    public static void main(String[] args) {
        BinomialTree tree = makeRandomTree(200, 200);
	    printTree(tree);

	    System.out.println("Is 5 in the tree?");
	    System.out.println(tree.search(5));
	    System.out.println("Is 7 in the tree?");
	    System.out.println(tree.search(7));
       	System.out.println("Is 18 in the tree?");
	    System.out.println(tree.search(18));

        System.out.println("the max is " + getMax(tree.getRoot()));
        System.out.println("the node height of max is " + getNodeHeight(tree.getRoot(), getMax(tree.getRoot()), 1));
        System.out.println("the max node height is " + getTreeHeight(tree.getRoot(), 1));
        System.out.println("the total number of nodes is " + getTotal(tree.getRoot(), 1));
        System.out.println("the number that is one higher than 1 is " + getNextHighest(tree.getRoot(), 1));
    }

}
