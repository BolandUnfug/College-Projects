import java.util.ArrayList;
import java.awt.*;
import java.awt.image.*;
import javax.imageio.*;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.time.YearMonth;
import java.util.Scanner;
import java.nio.file.Path;
import java.nio.file.Paths;

public class mazereader {
    public static void main(String[] args) throws Exception {
        boolean done = false;
        while (done == false) {
            BufferedImage maze = imagedownload();
            BufferedImage binerizedmaze = binerize(maze);
            BufferedImage heatmapcopy = binerize(maze);
            JFrame window = ImageFilter.displayImage(heatmapcopy, "FinalHeatmap");
            JFrame window1 = ImageFilter.displayImage(heatmapcopy, "FinalPath");
            int map[][] = nummap(binerizedmaze);
            int positions[][] = startandendfinder(map, heatmapcopy);
            int heatmaparray[][] = update(map, positions, heatmapcopy);
            int path[][] = solution(map, positions[0]);
            BufferedImage heatmap = MazeSolver_HelperFunctions.visualizeDistances(heatmapcopy, heatmaparray, window);
            imageupload(heatmap, "mazereader/Heatmap.png");
            BufferedImage solution = MazeSolver_HelperFunctions.visualizePath(binerizedmaze, path, window1);
            imageupload(solution, "mazereader/Solution.png");

            System.out.println("mapdone");
            Scanner input = new Scanner(System.in);
            System.out.println("end the program? y/n");
            String finish = input.nextLine();
            if (finish.equals("y")) {
                done = true;
            } else {
                done = false;
            }
        }
        System.out.println("Thanks for playing!");

    }

    public static void imageupload(BufferedImage mazecopy, String Filename) throws Exception {
        File outputfile = new File(Filename);
        ImageIO.write(mazecopy, "png", outputfile);
    }

    public static BufferedImage imagedownload() throws Exception {
        Scanner input = new Scanner(System.in);
        Path inPath = Path.of("mazereader/" + "1.png");
        System.out.println("Pick a maze from 1-7");
        int mazename = 1;
        BufferedImage maze;
        mazename = input.nextInt();
        maze = ImageIO.read(new File("mazereader/1.png"));
        if (mazename == 1) {
            maze = ImageIO.read(new File("mazereader/1.png"));
        }
        if (mazename == 2) {
            maze = ImageIO.read(new File("mazereader/2.png"));
        }
        if (mazename == 3) {
            maze = ImageIO.read(new File("mazereader/3.jpg"));
        }
        if (mazename == 4) {
            maze = ImageIO.read(new File("mazereader/4.png"));
        }
        if (mazename == 5) {
            maze = ImageIO.read(new File("mazereader/5.png"));
        }
        if (mazename == 6) {
            maze = ImageIO.read(new File("mazereader/6.png"));
        }
        if (mazename == 7) {
            maze = ImageIO.read(new File("mazereader/7.png"));
        }

        // ImageOutputStream
        // ImageIO.write(im, formatName, output)
        return maze;
    }

    public static int[][] nummap(BufferedImage maze) {
        int map[][] = new int[maze.getHeight()][maze.getWidth()];
        int rgb = 0;
        int tile = 0;
        for (int row = 0; row < maze.getHeight(); row++) {
            for (int col = 0; col < maze.getWidth(); col++) {
                rgb = maze.getRGB(col, row);
                // System.out.println(rgb);
                if (rgb == -16777216) {
                    tile = 1;
                } else if (rgb == -1) {
                    tile = 0;
                }

                map[row][col] = tile;
            }
        }
        return map;
    }

    public static int[][] startandendfinder(int map[][], BufferedImage maze) {
        int startpos[] = { 0, 0 };
        int endpos[] = { map.length - 1, map[0].length - 1 };
        /*
         * Scanner input = new Scanner(System.in);
         * System.out.print("Image Size: " + map.length + " x " + map[0].length);
         * System.out.print("Input coordinates? y/n");
         * String customcoords = input.nextLine();
         * if(customcoords.equals("y")){
         * System.out.println("Input a starting X coordinate:");
         * startpos[0] = input.nextInt();
         * System.out.println("Input a starting Y coordinate:");
         * startpos[1] = input.nextInt();
         * System.out.println("Input a ending X coordinate:");
         * endpos[0] = input.nextInt();
         * System.out.println("Input a ending Y coordinate:");
         * endpos[1] = input.nextInt();
         * map[endpos[0]][endpos[1]] = 2;
         * }
         * else{
         */
        int rgb, red, blue;
        Color pixel;
        for (int row = 0; row < map.length; row++) {
            for (int col = 0; col < map[0].length; col++) {
                rgb = maze.getRGB(col, row);
                pixel = new Color(rgb);
                red = pixel.getRed();
                blue = pixel.getBlue();
                if (red == 255 && blue != 255) {
                    // System.out.println("red check " + red);
                    endpos[0] = row;
                    endpos[1] = col;
                    map[row][col] = 2;
                }
                if (blue == 255 && red != 255) {
                    // System.out.println("blue check " + blue);
                    startpos[0] = row;
                    startpos[1] = col;
                    map[row][col] = 0;
                }

            }
        }
        int positions[][] = { startpos, endpos };
        System.out
                .println(positions[0][0] + " " + positions[0][1] + " " + positions[1][0] + " " + positions[1][1] + " ");
        return positions;
    }

    public static int[][] update(int map[][], int startandendpoints[][], BufferedImage maze) {
        System.out.println("Update trigger");
        JFrame window = ImageFilter.displayImage(maze, "title");
        // generating the nested list that controls which tiles to update
        ArrayList<ArrayList<Integer>> updates = new ArrayList<ArrayList<Integer>>();
        // initializing first set of coordinates
        updates.add(new ArrayList<Integer>());
        updates.get(0).add(startandendpoints[1][0]);
        updates.get(0).add(startandendpoints[1][1]);
        int coordinates[] = { updates.get(0).get(0), updates.get(0).get(1) };
        // System.out.println("starting coords: X:" + startandendpoints[0][0] + " Y:" +
        // startandendpoints[0][1]);
        // System.out.println("X: "+coordinates[0] +" Y: "+ coordinates[1]);
        double updaterate = 0;
        // setting the modifier for each tile
        double triggers = 0;
        if (maze.getWidth() >= 500 && maze.getWidth() < 600) {
            updaterate = 500;
        } else if (maze.getWidth() >= 700 && maze.getWidth() < 800) {
            updaterate = 2500;
        }

        else if (maze.getWidth() >= 1800 && maze.getWidth() < 1900) {
            updaterate = 6000;
        }

        else if (maze.getWidth() >= 3000) {
            updaterate = 40000;
        }
        while (updates.size() > 0) {

            triggers++;

            if (triggers >= updaterate) {
                triggers = 0;
                MazeSolver_HelperFunctions.visualizeDistances(maze, map, window);

            }

            // System.out.println("Update Loop trigger" + triggers);
            coordinates[0] = updates.get(0).get(0);
            coordinates[1] = updates.get(0).get(1);
            // System.out.println("number of tiles to update:" + updatesize);
            // extracting coordinates from nested list for simplification
            // System.out.println("");
            // System.out.println("X: "+coordinates[0] +" Y: "+ coordinates[1]);
            // delete the tile
            if (updates.size() > 0) {
                // System.out.println("Removed coordinate" + updates.get(0));
                updates.remove(0);
            }
            // System.out.println(coordinates[0]);
            // System.out.println(coordinates[1]);

            /* UP */
            if (coordinates[0] != 0) {
                // System.out.println("Up Trigger 1");
                // System.out.println(map[coordinates[0]-1][coordinates[1]]);
                if (map[coordinates[0] - 1][coordinates[1]] == 0) {
                    updates.add(new ArrayList<Integer>());
                    updates.get(updates.size() - 1).add(coordinates[0] - 1);
                    updates.get(updates.size() - 1).add(coordinates[1]);
                    map[coordinates[0] - 1][coordinates[1]] = map[coordinates[0]][coordinates[1]] + 1;
                    // System.out.println("Up Trigger 2");
                    // System.out.println("added choordinate " + updates.get(0));
                }
            }
            /* DOWN */
            if (coordinates[0] != map.length - 1) {
                // System.out.println("Down Trigger 1");
                // System.out.println(map[(coordinates[0]) + 1][coordinates[1]]);
                if (map[(coordinates[0]) + 1][coordinates[1]] == 0) {
                    updates.add(new ArrayList<Integer>());
                    updates.get(updates.size() - 1).add(coordinates[0] + 1);
                    updates.get(updates.size() - 1).add(coordinates[1]);
                    map[(coordinates[0]) + 1][coordinates[1]] += map[coordinates[0]][coordinates[1]] + 1;
                    // System.out.println("Down Trigger 2");
                    // System.out.println("added choordinate " + updates.get(0));
                }
            }
            /* LEFT */
            if (coordinates[1] != 0) {
                // System.out.println("Left Trigger 1");
                // System.out.println(map[coordinates[0]][(coordinates[1]) - 1]);
                if (map[coordinates[0]][(coordinates[1]) - 1] == 0) {
                    updates.add(new ArrayList<Integer>());
                    updates.get(updates.size() - 1).add(coordinates[0]);
                    updates.get(updates.size() - 1).add(coordinates[1] - 1);
                    map[coordinates[0]][(coordinates[1]) - 1] += map[coordinates[0]][coordinates[1]] + 1;
                    // System.out.println("Left Trigger 2");
                    // System.out.println("added choordinate " + updates.get(0));
                }
            }
            /* RIGHT */
            if (coordinates[1] != map[0].length - 1) {
                // System.out.println("Right Trigger 1");
                // System.out.println(map[coordinates[0]][(coordinates[1]) + 1]);
                if (map[coordinates[0]][(coordinates[1]) + 1] == 0) {
                    updates.add(new ArrayList<Integer>());
                    updates.get(updates.size() - 1).add(coordinates[0]);
                    updates.get(updates.size() - 1).add(coordinates[1] + 1);
                    map[coordinates[0]][(coordinates[1]) + 1] += map[coordinates[0]][coordinates[1]] + 1;
                    // System.out.println("Right Trigger 2");
                    // System.out.println("added choordinate " + updates.get(0));
                }
            }
            // print(map);

        }
        // System.out.println("X: "+coordinates[0] +" Y: "+ coordinates[1]);
        window.dispose();
        return map;
    }

    public static int[][] solution(int map[][], int startpos[]) {
        System.out.println("Solution Trigger");
        int coordinates[] = { startpos[0], startpos[1] };

        int coordinatepoints[][] = new int[map[startpos[0]][startpos[1]] + 1][2];
        // System.out.println(coordinatepoints.length);
        // System.out.println(coordinates[0] + " " + coordinates[1]);
        coordinatepoints[0][0] = coordinates[0];
        coordinatepoints[0][1] = coordinates[1];

        // System.out.println("mapvalue" + map[startpos[0]][startpos[1]]);
        for (int x = map[startpos[0]][startpos[1]]; x > 1; x--) {
            // System.out.println("Solution Loop Trigger " + x);
            // System.out.println(map[coordinates[0]][coordinates[1]]);
            /* UP */
            if (coordinates[0] != 0) {
                if (map[coordinates[0] - 1][coordinates[1]] == map[coordinates[0]][coordinates[1]] - 1
                        && map[coordinates[0] - 1][coordinates[1]] != 1) {
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][0] = coordinates[0];
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][1] = coordinates[1];
                    coordinates[0] = coordinates[0] - 1;
                    coordinates[1] = coordinates[1];
                    // System.out.println("up updated" + coordinates[0]);
                    // System.out.println("up updated" + coordinates[1]);
                }
            }
            /* DOWN */
            if (coordinates[0] != map.length - 1) {
                if (map[(coordinates[0]) + 1][coordinates[1]] == map[coordinates[0]][coordinates[1]] - 1
                        && map[coordinates[0] + 1][coordinates[1]] != 1) {
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][0] = coordinates[0];
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][1] = coordinates[1];
                    coordinates[0] = coordinates[0] + 1;
                    coordinates[1] = coordinates[1];

                    // System.out.println("down updated" +coordinates[0]);
                    // System.out.println("down updated" +coordinates[1]);
                }
            }
            /* LEFT */

            if (coordinates[1] != 0) {
                if (map[coordinates[0]][(coordinates[1]) - 1] == map[coordinates[0]][coordinates[1]] - 1
                        && map[coordinates[0]][coordinates[1] - 1] != 1) {
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][0] = coordinates[0];
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][1] = coordinates[1];
                    coordinates[0] = coordinates[0];
                    coordinates[1] = coordinates[1] - 1;
                    // System.out.println("left updated" +coordinates[0]);
                    // System.out.println("left updated" +coordinates[1]);
                }
            }

            /* RIGHT */
            if (coordinates[1] != map[0].length - 1) {
                if (map[coordinates[0]][(coordinates[1]) + 1] == map[coordinates[0]][coordinates[1]] - 1
                        && map[coordinates[0]][coordinates[1] + 1] != 1) {
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][0] = coordinates[0];
                    coordinatepoints[map[coordinates[0]][coordinates[1]]][1] = coordinates[1];
                    coordinates[0] = coordinates[0];
                    coordinates[1] = coordinates[1] + 1;
                }

                // System.out.println(coordinatepoints[map[coordinates[0]][coordinates[1]]][0] +
                // "" + coordinatepoints[map[coordinates[0]][coordinates[1]]][1]);
            }
        }
        return coordinatepoints;
    }

    public static void print(int map[][]) {
        for (int row = 0; row < map.length; row++) {
            for (int col = 0; col < map[0].length; col++) {
                System.out.print(map[row][col]);
            }
            System.out.println();
        }
    }

    public static BufferedImage binerize(BufferedImage maze) {
        BufferedImage mazecopy = ImageFilter.copy(maze);
        int rgb, red, green, blue;
        double intensity;
        Color pixel, colorout;
        boolean bluefound = false, redfound = false;
        for (int row = 0; row < maze.getHeight(); row++) {
            for (int col = 0; col < maze.getWidth(); col++) {
                rgb = maze.getRGB(col, row);
                pixel = new Color(rgb);
                red = pixel.getRed();
                red *= red;
                green = pixel.getGreen();
                green *= green;
                blue = pixel.getBlue();
                blue *= blue;
                intensity = Math.sqrt(red + green + blue);
                if (red > blue && red > green && red >= 250 && redfound == false) {
                    redfound = true;
                    // System.out.println("red trigger" + red);
                    colorout = new Color(255, 0, 0);
                    // System.out.println(colorout.getRed());
                } else if (blue > red && blue > green && blue >= 250 && bluefound == false) {
                    bluefound = true;
                    // System.out.println("blue trigger" + blue);
                    colorout = new Color(0, 0, 255);
                    // System.out.println(colorout.getBlue());
                } else {
                    if (intensity >= 350) {
                        colorout = new Color(255, 255, 255);
                    } else {
                        colorout = new Color(0, 0, 0);
                    }
                }
                mazecopy.setRGB(col, row, colorout.getRGB());
                //
            }
        }
        return mazecopy;
    }

}