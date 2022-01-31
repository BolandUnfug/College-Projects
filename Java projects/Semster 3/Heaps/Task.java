public class Task {
    private Comparable thing;
    private Comparable priority;
    private Task left;
    private Task right;
    private Task parent;

    public Comparable getThing() {
        return thing;
    }

    public void changeThing(Comparable thing) {
        this.thing = thing;
    }

    public Comparable getPriority() {
        return priority;
    }

    public void changePriority(Comparable priority) {
        this.priority = priority;
    }

    public Task getLeft() {
        return left;
    }

    public void setLeft(Task nd) {
        left = nd;
    }

    public Task getRight() {
        return right;
    }

    public void setRight(Task nd) {
        right = nd;
    }

    public Task getParent() {
        return parent;
    }

    public void setParent(Task nd) {
        parent = nd;
    }

    Task() {

    }

    Task(Comparable thing) {
        this.thing = thing;
        parent = null;
        left = null;
        right = null;
    }

    Task(Comparable thing, Comparable priority) {
        this.thing = thing;
        this.priority = priority;
        parent = null;
        left = null;
        right = null;
    }
}