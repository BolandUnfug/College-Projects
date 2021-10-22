public class TNode {
    //This class represents a TNode in a Linked List
    //The class holds one character of data
    private Comparable data;
    private TNode above;
    private TNode nextgreater;
    private TNode nextlesser;

    public TNode(Comparable data){
	this.data = data;
    }

    public Comparable getData(){
	return data;
    }

    public TNode getAbove(){
    return above;
    }

    public TNode getNextGreater(){
	return nextgreater;
    }

    public TNode getNextLesser(){
    return nextgreater;
    }

    public void setData(char c){
	data = c;
    }
    
    public void setAbove(TNode n){
    above = n;
    }

    public void setNextGreater(TNode n){
    nextgreater = n;
    }

    public void setNextLesser(TNode n){
	nextlesser = n;
    }
}