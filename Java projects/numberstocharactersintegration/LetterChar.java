/**
 * LetterChar, implements the Letter interface and creates ASCII characters stored in letter form.
 * <br>
 * can get either the ASCII number or the letter through getInt or getChar.
 * @author Boland Unfug, 8/24/2021
*/
public class LetterChar implements Letter {
    private char numletter;
    LetterChar(char numletter){
        int letternum = (int)numletter;
        if((letternum >=65 && letternum <=90) || (letternum >=97 && letternum <=122)){
            this.numletter = numletter;
        }
        else{        
            throw new RuntimeException();
        }
        
    }
    public char getChar(){
        return numletter;
    }

    public int getInt(){
        return (int) numletter;
    }
    
}
