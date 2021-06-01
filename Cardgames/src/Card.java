/**
 * manages the card object
 * @author Boland Unfug, New College of Florida, 5/26/2021
 */
public class Card {
    //Fields
    int cardvalue;
    char suite;
    /**
     * Card constructor, creates a card object
     * @param cardvalue int, the type of card, ie ace,5, 7
     * @param suite char, the suite of the card, ie spades and hearts
     */
    Card(int cardvalue, char suite){
        this.cardvalue = cardvalue;
        this.suite = suite;
    }
}
