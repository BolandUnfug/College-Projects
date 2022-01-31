//IMPORTANT: You will modify this class to add functions
//the functions that you need to add are described in the assignment document

//All functions in this class are static.
//This class is never intended to be constructed.

/**
 * Tree Utilities provides some usefull functions for the tree data structure.
 * Boland Unfug, got the recursive idea from fernie
 * 
 */
import java.util.Random;

public class GoatTreeUtilities{

	public static int getMax(SGNode nd){
		return nd.getRight() != null ? getMax(nd.getRight()) : nd.getData();
	}

	public static int getTotal(SGNode nd, int counter){ // to be added later for fun, counts the number of items  in the tree
		int temp1 = 0;
		int temp2 = 0;
		if(nd.getLeft() != null){
			temp1 = getTotal(nd.getLeft(), counter);
		}
		if (nd.getRight() != null){
			temp2 = getTotal(nd.getRight(), counter);
		}
		return temp1 + temp2 + counter;
	}

	public static int getNodeHeight(SGNode nd, int target,int counter){
		if( nd.getData() == target ){
			return counter;
		}
		
		if( target > nd.getData() ){
			//Find larger values on the right
			if( nd.getRight() != null){
			return getNodeHeight(nd.getRight(), target, counter + 1);
			}
		}
	
		else if( target < nd.getData()){
			//Find smaller values on the left
			if(nd.getLeft() != null){
			return getNodeHeight(nd.getLeft(), target, counter + 1);
			}
		}
		return counter;
	}

	public static int getTreeHeight(SGNode nd, int counter){
		if(nd != null){
			counter++;
			int temp1 = getTreeHeight(nd.getLeft(), counter);
			int temp2 = getTreeHeight(nd.getRight(), counter);
			counter = temp1 > temp2 ? temp1 : temp2;
		}
		return counter;
	}

	public static int getNextHighest(SGNode nd, int target){
		System.out.println("looking for " + target + " is it in the tree? " + nd.search(target + 1));
		return nd.search(target + 1) == false ? getNextHighest(nd, target + 1) : nd.nodeSearch(target+1).getData();
	}
	

    public static void printTreeHelper(SGNode nd){
	if (nd != null){
	    printTreeHelper(nd.getLeft());
	    System.out.println("data: " + nd.getData() + " height: " + nd.getHeight() + " descendants: " + nd.getDescendants());
		
	    printTreeHelper(nd.getRight());
	}
    }

    //This function prints all of the data an ScapeGoatTree in order
    //Starting with the lowest and going to the highest
    public static void printTree(ScapeGoatTree tree){
	printTreeHelper(tree.getRoot());
	System.out.println("");
    }

    //This function creates a ScapeGoatTree filled with randomly chosen data
    //num is the number of nodes in the random search tree
    //max_val is the upper limit on the randomly chosen integer data
    public static ScapeGoatTree makeRandomTree(int num, int max_val){
	Random rand = new Random();
	ScapeGoatTree tree = new ScapeGoatTree();
	
	for(int i=0; i<num; i++){
	    int rand_data = rand.nextInt(max_val);
	    tree.insert(rand_data);
	}

	return tree;
    }

}
