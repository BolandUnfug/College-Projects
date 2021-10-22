import java.util.Random;
public class TestTree {
    public static void main(String[] args) {
        Random rand = new Random();
        BinaryTree tree = new BinaryTree();
        for(int i = 0; i < 10; i++){
            tree.insert(rand.nextInt(10));
        }
        tree.print(tree.start);
        System.out.println(tree.getHeight(tree.start, 0));
    }
}
