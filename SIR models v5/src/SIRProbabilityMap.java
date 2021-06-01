
/**
 *  Defines the map used in the SIR model: object-oriented implementation for a kernel distribution <br>
 * <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */


import java.util.Random;
import java.lang.Math;

public class SIRProbabilityMap extends SIRMap{
    private final int seednum;

    public SIRProbabilityMap(int size) {
        this.size = size;
        System.out.println("trigger");
        this.seednum = 4; //sets the number of seeds to 4
        this.population = new SIRCell[size][size]; // references are initially null
    }

    /**
     * Randomly initialize the map's population, using kernel distributions.
     * @param density double, the percent of occupied cells
     * @param maskdensity double, the percent of occupied cells
     * @param seeds int, takes seeds but actually does nothing
     */
    public void populate(double density, double maskdensity, int seeds) {
        Random rand = new Random();
        SIRState startingstate = SIRState.SUSCEPTIBLE; //sets base case to SUSCEPTIBLE
        double probabilities[];
        int [][] seedCoordinates = newseedCoords(); // generates a new set of seed coordinates
        for(int a = 0; a < 4; a++){                     //prints seed coordinates
            System.out.println(seedCoordinates[a][0]);
            System.out.println(seedCoordinates[a][1]);
        }
        for (int row = 0; row < this.size; row++) {
            for (int col = 0; col < this.size; col++) {
                
                probabilities = getProbabilities(row, col, seedCoordinates); // gets the probabilities of the cell, based on its position relative to the seeds

                double occupancy = rand.nextDouble(); // sets a random amount of cells to infectious to start the model
                if (occupancy < density) {
                    startingstate = SIRState.INFECTIOUS;
                }
                else{
                    startingstate = SIRState.SUSCEPTIBLE;
                }
                double masks = rand.nextDouble(); // decides who is going to be what based on the above proportions
                    masks = rand.nextDouble();
                    if(masks <= probabilities[0]){
                        this.population[row][col] = new MaskedSIRCell(startingstate);
                    }
                    else if (masks <= probabilities[1]+probabilities[0]){ // by adding each one to the other, I measure their percentage instead of the weight
                        this.population[row][col] = new QuarintineSIRCell(startingstate);
                    }
                    else if (masks <= probabilities[2]+probabilities[1]+probabilities[0]){
                        this.population[row][col] = new SuperSafeSIRCell(startingstate);
                    }
                    else if (masks <= probabilities[3]+probabilities[2]+probabilities[1]+probabilities[0]){
                        this.population[row][col] = new UnMaskedSIRCell(startingstate);
                    }
                    else{
                        this.population[row][col] = new MaskedSIRCell(startingstate); // because of proportions, its not always perfectly accurate so I need overflow
                        System.out.println("BROKEd");
                        System.out.println("mask: " + masks);
                        System.out.println("probabilities:");
                        System.out.println("1:" +probabilities[0]);
                        System.out.println("2:" +probabilities[1]);
                        System.out.println("3:" +probabilities[2]);
                        System.out.println("4:" +probabilities[3]);
                    }
                
            }
        }
        formNeighborhoods(); // god why did this have so many problems. gives every cell neighbors
    }

    /**
     * generate 4 random coordinates for the seeds.
     * @return int[][], an array of seed coordinates. in the future I will make this dynamic
     */
    private int[][] newseedCoords(){
        Random rand = new Random();
        int[][] seedCoords = new int[seednum][2];
        for(int i = 0; i < seednum; i++){
            seedCoords[i][0] = rand.nextInt(size);
            seedCoords[i][1] = rand.nextInt(size);
        }
        return seedCoords;
    }

    /**
     * Calculates the distance between 2 points.
     * @param int x1, the first coordinates x point
     * @param int x2, the second coordinates x point
     * @param int y1, the first coordinates y point
     * @param int y2, the second coordinates y point
     * @return double distance. applies the pythagorian theorem to get a length from two points by making a triangle.
     */
    private double getDistance(int x1, int x2, int y1, int y2){
        return Math.sqrt(Math.pow((x1-x2), 2) + Math.pow(((y1 - y2)), 2));
    }

    /**
     * calculates the distance between 2 1 dimensional points, ie either 2 x coordinates or 2 y coordinates.
     * @param int num1, the first coordinate
     * @param int num2, the second coordinate
     * @return double distance. gets the absolute difference between the two points.
     */
    private double getDistance(int num1, int num2){
        return Math.abs(num1-num2);
    }

    /**
     * The worlds most complicated distribution method. why did I even try to use this lol. Also, it can be improved and then used at a later date.
     * @param double xdistance, takes the distance between two points on the x axis.
     * @param double ydistance, takes the distance between two points on the y axis.
     * @return double result, the "weight" given to the cell from a specefic kernel based on its proximity. the weight is similar to a probability.
     */
    private double gaussianWeight(double xdistance, double ydistance){

        double stddev = this.size/2; // distance of one deviation
        double part1 = 1/(2*Math.PI*Math.pow(stddev, 2));
        double ePowFraction = (Math.pow(xdistance,2)+Math.pow(ydistance,2))/(2*Math.pow(stddev, 2));
        double e = Math.pow(Math.E, - (ePowFraction));
        double result = part1*e;
        //System.out.println("result:" + result);
        return result;
    }


    /**
     * This is my sample distribution, which is just a basic curve that I took a portion of.
     * @param double distance, takes the distance between two points.
     * @return double result, the "weight" given to the cell from a specefic kernel based on its proximity. the weight is similar to a probability.
     */
    private double ExponentWeight(double distance){

        double stddev = this.size/2; // distance of one deviation
        double x = distance/stddev;
        double result = 1/(Math.pow(x, 2));
        return result;
    }
    /**
     * The worlds most complicated distribution method. why did I even try to use this lol. Also, it can be improved and then used at a later date.
     * @param int cellx, the current cells x position.
     * @param int celly, the current cells y position.
     * @param int[][] seeds, an array of seed coordinates. used to calculate the distance to each of the seeds.
     * @return double[] probabilities, an array of probabilities that add up to roughly 1. these probabilities decide what type of cell it will become.
     */
    private double[] getProbabilities(int cellx,int celly, int[][] seeds){
        int seednum = seeds.length; // I made this a variable because I thought I would use it more
        double[] weights = new double[seednum]; // there are equal probabilities to the amount of seeds, meaning I dont have to dynamically make space
        double[] probabilities = new double[seednum];
        double chancemultiplier = 0; // the variable in charge of making the probabilities add up to 1
        double weight;
        for(int i = 0; i < seednum; i++){ // iterates for each seed
            
            weight = ExponentWeight(getDistance(cellx, seeds[i][0], celly, seeds[i][1])); // gets the weight of a cell from seed i
            weights[i] = weight; // add the weight to the list
            chancemultiplier += weight; // add the weight to the total
        }
        for(int j = 0; j < seednum; j++){
            probabilities[j] = weights[j]/chancemultiplier; //by dividing by the total of the weights, you essentially divide by itself to end up with 1
        }
        
        return probabilities;
    }   
}