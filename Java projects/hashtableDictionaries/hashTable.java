import java.util.ArrayList;
import java.util.Iterator;

public class hashTable implements Dictionary{
    private int size;
    protected ArrayList<ArrayList<keyNode>> list;

     public hashTable(){
         list = new ArrayList<>(1000);
         size = 1000;
         for(int i = 0; i < size; i++){
            ArrayList<keyNode> sublist = new ArrayList<>();
            list.add(sublist);
         }
     }

    public void inserting(String key, Comparable value) {
        // creates a new keynode with the above values and creates a hashcode
        //size++;
        
        keyNode data = new keyNode<Comparable>(key, value);
        
        int hashed = key.hashCode() % size;
        if(hashed < 0){
            hashed = hashed - (hashed * 2);
        }
        System.out.println(hashed);
        list.get(hashed).add(data);
    }

    @Override
    public Object lookup(String key) {
        int hashed = key.hashCode()%size;
        if(hashed < 0){
            hashed = hashed - (hashed*2);
        }
        return list.get(hashed).get(0).getValue();
    }

    @Override
    public void insert(String key, Object value) {
        System.out.println("inserting " + value);
        inserting(key,value.toString());
        //resize();
    }

    // public void resize(){
    //     // moving everything over to a new arraylist
    //     // go through all of the previous options
    //     // add them to the new list
    //     // set the old list equal to the previous list
    //     //
    //     //
    //     //




    //     ArrayList<ArrayList<keyNode>> newlist = new ArrayList<>();
    //     for(int i = 0; i < size; i++){
    //         ArrayList<keyNode> sub = new ArrayList<>();
    //         newlist.add(sub);
    //     }
    //     Iterator<ArrayList<keyNode>> listiter = list.iterator();
    //     int i = 0;
    //     while(listiter.hasNext() == true){
    //         System.out.print("first loop, i = " + i);
    //         int j = 0;
    //         Iterator<keyNode> sublistiter = list.get(i).iterator();
    //         i++;
    //         while(sublistiter.hasNext() == true){
    //             System.out.print("second loop, j = " + j);
    //             System.out.println();
    //             keyNode temp = list.get(i).get(j);
    //             int hashed = temp.getKey().hashCode() % size;
    //             if(newlist.get(i) == null){
    //                 ArrayList<keyNode> sublist = new ArrayList<>();
    //                 sublist.add(temp);
    //                 newlist.add(hashed, sublist);
    //             }
                
    //             if(newlist.get(i).get(j) != null){
    //                 newlist.get(hashed).add(temp);
    //             }
                
    //         }
    //     }
    //     list = newlist;

    // }

    
}
