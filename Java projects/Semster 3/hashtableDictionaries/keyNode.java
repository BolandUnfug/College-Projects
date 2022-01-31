public class keyNode <Comparable>{
    private String key;
    private Comparable value;

    public keyNode(String key, Comparable value){
        this.key = key;
        this.value = value;
    }

    public Comparable getValue(){
        return value;
    }

    public String getKey(){
        return key;
    }
}
