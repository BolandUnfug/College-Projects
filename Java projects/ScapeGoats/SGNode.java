import java.util.ArrayList;
import java.util.Collections;

//This is a binary search tree that stores integers

public class SGNode{
    public SGNode left;
    public SGNode right;
    public SGNode parent;
    public int data;
	public int height;
	public int descendants;

	
			
	/**
	 * works just like the insert function, but also keeps track of each nodes height, and the number of descendants
	 * @param new_data the data to be inserted
	 * @param counter the starting height, has to be 0
	 */
    public void insert(int new_data, int counter){
		//System.out.println("inserting " + new_data + " at level " + counter);
		this.descendants++; // descendants works, just need to fix height

	if( new_data >= data){
	    //Insert larger values on the right
	    if( right == null){
		this.height = counter;
		right = new SGNode(new_data);
		right.setParent(this);
	    }
	    else{
		right.insert(new_data, counter);
	    }
	}

	else if( new_data < data){
	    //Insert smaller values on the left
		//System.out.println("moved to the left.");
	    if( left == null){
		this.height = counter;
		left = new SGNode(new_data);
		left.setParent(this);
	    }
	    else{
		left.insert(new_data, counter);
	    }
	}
    }

    public boolean search(int target){
	if( data == target ){
	    return true;
	}
	if( target > data ){
	    //Find larger values on the right
	    if( right != null){
		return right.search(target);
	    }
	}

	if( target < data){
	    //Find smaller values on the left
	    if(left != null){
		return left.search(target);
	    }
	}
	return false;
    }

	public SGNode nodeSearch(int target){
		if( data == target ){
			return this;
		}
		if( target > data ){
			//Find larger values on the right
			if( right != null){
			return right.nodeSearch(target);
			}
		}
	
		if( target < data){
			//Find smaller values on the left
			if(left != null){
			return left.nodeSearch(target);
			}
		}
		
		return null;
		}

    public int getData(){
	return data;
    }

    public void setData(int d){
	data = d;
    }

    public SGNode getLeft(){
	return left;
    }

    public void setLeft(SGNode nd){
	left = nd;
    }

    public SGNode getRight(){
	return right;
    }

    public void setRight(SGNode nd){
	right = nd;
    }

    public SGNode getParent(){
	return parent;
    }

    public void setParent(SGNode nd){
	parent = nd;
    }

    public boolean isLeaf(){
	if ( (left == null) && (right == null) ){
	    return true;
	}
	else {
	    return false;
	}
    }

	public void setHeight(int height){
		this.height = height;
	}

	public void setDescendants(int descendants){
		this.descendants = descendants;
	}

	public int getHeight(){
		return this.height;
	}

	public int getDescendants(){
		return this.descendants;
	}

    //Constructor
    public SGNode(int the_data){
	data = the_data;
	height = 0;
	descendants = 0;
	left = null;
	right = null;
    }
}