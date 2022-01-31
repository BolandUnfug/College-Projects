import java.util.ArrayList;
import java.util.Collections;


public class ScapeGoatTree implements IntDataSet{
    SGNode root;


    /**
	 * This inserts a node into the scape goat tree. currently does not work with duplicates.
	 * @param new_data the data to be inserted
	 * @param counter the starting height of the node, needs to be 0
	 */
    @Override
	public void insert(int new_data){
        //System.out.println("Adding " + new_data + " to the tree");
        if(root == null){
           // System.out.println("needed a starter  node.");
            root = new SGNode(new_data); 
        }
        else{
        int counter = 0;
		root.insert(new_data, counter);
		SGNode scapegoat = findScapeGoat(root.nodeSearch(new_data));
		if(scapegoat != null){
			fixGoat(scapegoat);
		}
    }
		// balances tree below the scapegoat
	}

    /**
	 * locates a scapegoat. works from the bottom up.
	 * @param node the node being searched, moves up each iteration
	 * @return the scapegoat node, or null if no scapegoat was found
	 */
	public SGNode findScapeGoat(SGNode node){
		if(node != null){
			return 1.5 * node.getHeight() > node.getDescendants() ? node : findScapeGoat(node.getParent());
		}
		//System.out.println("none found.");
		return null;
	}

    /**
     * fixes a scapegoat tree. 
     * @param node the scapegoat node, in which it and everything below it will be reshaped.
     */
    public void fixGoat(SGNode node){
        //System.out.println("need to fix a node.");
		ArrayList<Integer> list = new ArrayList<>();
		ArrayList<Integer> scapegoatlist = treeToList(list, node);
		Collections.shuffle(list);

		for(int i = 0; i < scapegoatlist.size(); i++){
			node.insert(scapegoatlist.get(i), node.getHeight());
		}		
	}

    /**
     * Moves a tree to a list
     * @param list the list being filled
     * @param node the starting node
     * @return a filled arraylist
     */
	public ArrayList<Integer> treeToList(ArrayList<Integer> list,SGNode node ){
		if(node != null){
			list.add(node.getData());
			treeToList(list, node.getLeft());
			treeToList(list, node.getRight());
		}
		return list;
	}

    public boolean search(int target){
        if( root == null){
            return false;
        }
        else{
            return root.search(target);
        }
        }
    
        public SGNode getRoot(){
        return root;
        }
    
        //Constructor
        public ScapeGoatTree(){
        root = null;
        }

}
