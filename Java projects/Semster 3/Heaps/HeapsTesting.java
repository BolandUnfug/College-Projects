import java.util.Random;
public class HeapsTesting {
    public static void main(String[] args) {
        int range = 100;
        GenericMaxHeap heap = new GenericMaxHeap<>();
        Random rand = new Random();
        System.out.println("Inserting:");
        for(int i = 0; i < range; i++){
            int inserted = rand.nextInt(range);
            heap.insert(inserted);
            System.out.println(inserted);
        }
        heap.print();
        
        System.out.println("Removing:");
        for(int i = 0; i < range; i++){
            //System.out.println("in loop");
            Comparable removed = heap.removeMax();
            System.out.println("Removed: " + removed);
            
        }
    }
}
/**
Add a bunch of Tasks to the heap with a variety of different priorities
Note: Do not insert the items in sorted order
Repeatedly remove the maximum priority task from the heap
Print out the removed task to verify that they are leaving the heap in priority order
 */