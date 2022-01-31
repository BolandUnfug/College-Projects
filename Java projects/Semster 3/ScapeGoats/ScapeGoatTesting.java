import java.util.ArrayList;
import java.util.Collections;

import javax.swing.RepaintManager;

public class ScapeGoatTesting extends GoatTreeUtilities{
    public static void main(String[] args) {

        int size = 10;
        int repetitions = 1;
        int heightaverage = 0;
        ScapeGoatTree tree = new ScapeGoatTree();
        for(int i = 0; i < repetitions; i++){
            ArrayList<Integer> list = generateRandomTree(size);
            for(int j = 0; j < size; j++){
                tree.insert(list.get(j));
            }
            heightaverage += getTreeHeight(tree.getRoot(), 0);
            System.out.println("tree height should not be 10: " + getTreeHeight(tree.getRoot(), 0));
        }
        heightaverage /= repetitions;
        printTree(tree);
        System.out.println("average length is: " + heightaverage);
    }

    public static ArrayList<Integer> generateRandomTree(int size){
        ArrayList<Integer> list = new ArrayList<>();
        for(int i = 0; i < size; i++){
            list.add(i);
        }
        Collections.shuffle(list);
        return list;
    }
}
