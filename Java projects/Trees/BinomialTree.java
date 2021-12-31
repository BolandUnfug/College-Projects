//This is a binary search tree that stores integers

public class BinomialTree{
    public TNode root;

    public void insert(int data){
	if( root == null ){
	    root = new TNode(data);
	}
	else{
	    root.insert(data);
	}
    }

    public boolean search(int target){
	if( root == null){
	    return false;
	}
	else{
	    return root.search(target);
	}
    }

    public TNode getRoot(){
	return root;
    }

    //Constructor
    public BinomialTree(){
	root = null;
    }

	
}
