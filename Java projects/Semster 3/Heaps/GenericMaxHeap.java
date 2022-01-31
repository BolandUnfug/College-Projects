//This class implements a heap (priority queue)
// ... The heap can store any comparable object
//For convenience we use an ArrayList
// ... The ArrayList implementation handles the re-sizing of the array
// ... which simplifies our code a bit

//This is a MaxHeap which means that the largest value is stored in the root
//... and the value at a node is Larger than the values of its children

import java.util.ArrayList;

public class GenericMaxHeap<T extends Comparable>{

    ArrayList<T> data;
    int size;

    //These functions take in an index (a location in the array)
    //... and return the index (location) of the left/right child
    int left(int index) {
        return 2*index + 1;
    }
    int right(int index) {
        return 2*index + 2;
    }

    //This functions take in an index (a location in the array)
    //... and return the index (location) of the parent
    int parent(int index) {
        return (index-1)/2;
    }

    //This function takes 
    public void insert(T new_data){
        int slot = data.size();
        //System.out.println(slot);
        data.add(new_data);
        //this.print();
        while(data.get(parent(slot)).compareTo(data.get(slot)) > 0  ){
            T temp = data.get(slot);
            data.set(slot, data.get(parent(slot)));
            data.set(parent(slot), temp);
            slot = parent(slot);
            System.out.println("Swapping " + data.get(parent(slot)) + " with " + data.get(slot));
        }
        size++;
    }

    //This function returns the largest item in the heap
    //This function does not change the heap
    public T getMax(){
        T largest = data.get(0);
        //System.out.println(size);
        for(int i = size; i >= size/2 ; i--){
            if(data.get(i).compareTo(largest) > 0){
                largest = data.get(i);
                //System.out.print(largest + " ");
            }
        }
        return largest;
    }

    //This function removes the largest item from the heap
    //This function then returns that item
    public T removeMax(){
    size--;
    //System.out.println("removing...");
    T thingremove = getMax();
    //System.out.print(thingremove);
    //System.out.println("found max.");
    int slot = data.indexOf(thingremove);
    data.set(slot, data.get(size));
    //System.out.println("found slot");
    data.remove(size);
	return thingremove;
    }

    public void print(){
        for(int i =0; i < this.size; i++){
            System.out.print(data.get(i) + " ");
        }
        System.out.println();
    }

    public GenericMaxHeap(){
	size = 0;
	data = new ArrayList<T>();
    }
}