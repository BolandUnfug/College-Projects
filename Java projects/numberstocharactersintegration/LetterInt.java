/**
 * LetterInt, implements the Letter interface and creates ASCII characters stored in integer form.
 * <br>
 * can get either the ASCII number or the letter through getInt or getChar.
 * @author Boland Unfug, 8/24/2021
*/
public class LetterInt implements Letter {
    private int letternum;
    LetterInt(int letternum){
        if((letternum >=65 && letternum <=90) || (letternum >=97 && letternum <=122)){
            this.letternum = letternum;
        }
        else{        
            throw new RuntimeException();
        }
    }
    public int getInt(){
        return letternum;
    }
    public char getChar(){
        return (char) letternum;
    }
    

}
