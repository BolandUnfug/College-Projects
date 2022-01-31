public class SNode<T extends Comparable> {
    private T data;
    private SNode<T> next;
    private SNode<T> previous;
    private SNode<T> below;
    private SNode<T> above;

    // constructor options

    SNode(T data){
        this.data = data;
    }

    SNode(SNode next){
        this.next = next;
    }

    SNode(SNode previous, SNode next){
        this.previous = previous;
        this.next = next;
    }

    //------------------------------------

    //sets each value

    public void setNext(SNode next){
        this.next = next;

    }

    public void setPrevious(SNode previous){
        this.previous = previous;

    }

    public void setAbove(SNode above){
        this.above = above;

    }

    public void setBelow(SNode below){
        this.below = below;

    }

    public void setData(T data){
        this.data = data;
    }

    //------------------------------------

    //get functions
    public SNode getNext(){
        return this.next;
    }

    public SNode getPrevious(){
        return this.previous;
    }
    
    public SNode getAbove(){
        return this.above;
    }

    public SNode getBelow(){
        return this.below;
    }

    public T getData(){
        return this.data;
    }

}
