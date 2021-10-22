import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.io.File;
import java.io.FilenameFilter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.file.DirectoryStream;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 *  Defines a CSV writing tool useful for a Susceptible-Infectious-Recovered (SIR) epidemiological model
 *  of viral spread, in the style of a cellular automaton (CA). Uses the SIR model's characteristics
 *  (infectionRate, recoveryRate, maxDays, gridSize) to autogenerate descriptive filenames.<br>
 *  <br>
 *  Not meant to be executed directly. Run SIR.java to test.
 *  <br>
 *  Compilation:  javac SIRWriter.java <br>
 *  Execution:    N/A   <br>
 *  <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
public class SIRWriter {

    private BufferedWriter writer;  // stream for writing to the output CSV file
    private String fileName;        // name of the output CSV file, defaults to: SIR_infectionRate*1000_recoveryRate*1000_maxDays_gridSize.csv
    private Path outPath;           // path to the output CSV file
    private String header;          // first row written to the output CSV file, explaining the name of each column
    private int filenum = 0;            // counts the number of files created

    /**
     * Fields are initialized in open(), when the SIR model is able to pass in its characteristics.
     */
    public SIRWriter(){
        writer = null;
        fileName = "";
        outPath = null;
        header = "";
    }

    /**
     * In case the user would like to dictate a specific file name.
     * @param fileName (String) a user-specified name for the output CSV file.
     */
    public SIRWriter( String fileName ){
        writer = null;
        this.fileName = fileName;
        System.out.println(fileName);
        outPath = null;
        header = "";
    }

    /**
     * Create an output stream for recording an SIR model's daily demographics.
     * @param maxDays (int) maximum allowed duration of the simulation, in days.
     * @param infectionRate (double) the daily probability that an infectious individual will infect each susceptible neighbor
     * @param recoveryRate (double) the daily probability that an infectious individual will recover
     * @param size (int) number of individuals along each side of the "city" city
     */
    public void open( int maxDays, double infectionRate, double recoveryRate, int size ){
        // Use the SIR model's fields to make the filename help users organize the results across trials
        // here is the error rn, invalid file path
        
        File results = new File("./CSVfiles");
        System.out.println("is it a directory?" + results.isDirectory());
        filenum = new File("./Java projects/SIR models v5/CSVfiles").listFiles().length;
        System.out.println(results);
        boolean fileexists = results.exists();
        System.out.println(fileexists);
         

        if (fileName.length() == 0) {
            fileName = String.format("SIR_%03d_%03d_%03d_%03d_%03d.csv", (int) (infectionRate * 1000), (int) (recoveryRate * 1000), maxDays, size, filenum);
        }
        
        outPath = Path.of(String.format("./Java projects/SIR models v5/CSVfiles/%s", fileName));
        System.out.println(outPath);

        // Start the output file with a header line at the top, which names each column of data
        SIRState state;
        try {
            BufferedWriter writer = Files.newBufferedWriter(outPath, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
            //BufferedWriter writer2  = Files.newBufferedWriter(outPath + fileName, cs, options)
            header = "Day";
            for( int i=0; i<SIRState.values().length; i++){
                state = SIRState.values()[ i ];
                header += String.format(",%s (total),%s (%%)", state, state);
            }
            header += "\n";
            writer.write( header );
            System.out.println( "Output file name: " + fileName );
        } catch(IOException e) {
            writer = null;
            System.out.println(e);
            System.out.println( "WARNING: SIRWriter unable to create output file." );
        }
    }

    /**
     * Record the day's results in the output file.
     * @param day (int) the current day in the simulation
     * @param totals (int[]) number of cells in each CA State
     * @param percentages (double[]) percent of cells in each CA State
     */
    public void update( int day, int[] totals, double[] percentages ){
        if( writer != null ) {
            String stats = "" + day;
            for( int state = 0; state < SIRState.values().length; state++ ){
                stats += "," + totals[state] + "," + percentages[state];
            }
            stats += "\n";
            try {
                writer.write( stats );
            } catch(IOException e){
                System.out.println("WARNING: SIRWriter failed to record day " + day );
            }
        }
    }

    /**
     * Safely close the output file.
     */
    public void close(){
        if (writer != null ) {  // The file can only be closed if it was successfully opened
            try { writer.close(); }
            catch (IOException e) {
                System.out.println( "WARNING: SIRWriter Failed to close the output stream." );
            }
        }
    }

}

