//Author: Yasser Elsayed

package ch.epfl.advdb;

import java.io.IOException;
import java.util.*;
import java.text.*;
import java.io.*;
import java.net.URI;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.filecache.*;

@SuppressWarnings("deprecation")
public class Main {

    public static int NUM_MAPPERS = 88;
    public static int NUM_REDUCERS = 88;
    public static int NUM_TOTAL_WORKERS = NUM_MAPPERS + NUM_REDUCERS;

    public static int MAX_ITERATIONS = 10;
    public static int MAX_SUB_ITER = 1;

    public static float dummyVal = -1e2F;
    public static float initialVal = 0.1F;
    public static int nRows = 480189;
    public static int nCols = 17770;
    public static int d = 10;
    public static String uMatrixFile = "U";
    public static String vMatrixFile = "V";
    public static String homeDir = "output";

    public static String uPath = "/U_0", vPath = "/V_0";
    public static long startTime, endTime;
    public static float duration;

    // an unchecked exception, to expose other exceptions that are caught
    public static class myException extends RuntimeException {
        public myException(String msg) {
            super(msg);
        }
        public myException() {
            super();
        }
    }
    public static void cout(Object o) {
        System.out.print(o);
    }

    // load a matrix from disk to a 2D array
    public static void buildMatrix(float[][] matrix, String path) {
        try {
            FileSystem fs = FileSystem.get(new Configuration());

            FileStatus[] files = fs.listStatus(new Path(path));
            for (FileStatus f : files) {
                if (new File(f.getPath().toString()).getName().startsWith("_"))
                    continue;
                cout("reading file: " + f.getPath().toString() + "\n");
                BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(f.getPath())));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] toks = line.split("\\s");
                    int r = Integer.parseInt(toks[1]);
                    int c = Integer.parseInt(toks[2]);
                    matrix[r][c] = Float.parseFloat(toks[3]);
                }
                br.close();
            }

        }
        catch (Exception e) {
            cout("EXCEPTION HAPPENED:\n" + e + "\n"
                    + Arrays.toString(e.getStackTrace()) + "\n");
            throw new myException(Arrays.toString(e.getStackTrace()));
        }
    }

    // ------------------------------------------------------------------------------------

    // this mapper strips out the last column, and emits a tuple by
    // userId where value is movieId,rating
    public static class PreProcessMap extends MapReduceBase implements
        Mapper<LongWritable, Text, Text, Text> {

            private Text zero = new Text("0");
            private Text keyText = new Text();
            private Text valText = new Text();

            public void map(LongWritable key, Text value,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {
                String[] toks = value.toString().split(",");
                keyText.set(toks[0]);
                valText.set(toks[1] + " " + toks[2]);
                output.collect(keyText, valText);
            }
        }

    // reducers collect movie ratings for each user, and emit one row
    // per user that contains normalized ratings for all nCols movies
    public static class PreProcessReduce extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            private Text valText = new Text();

            public void reduce(Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                float sum = 0.0F;
                ArrayList<Integer[]> receivedVals = new ArrayList();

                while (values.hasNext()) {
                    String[] pair = values.next().toString().split("\\s");
                    int i1 = Integer.parseInt(pair[0]);
                    int i2 = Integer.parseInt(pair[1]);
                    sum += i2;
                    receivedVals.add(new Integer[] {i1, i2});
                }
                float norm = sum / receivedVals.size();
                for (Integer[] pair : receivedVals) {
                    valText.set("M " + key.toString() + " " + pair[0] + " " + (pair[1] - norm));
                    output.collect(null, valText);
                }

            }
        }

    // ------------------------------------------------------------------------------------


    // mapper emits one row per U or V element with the squared difference
    public static class InitRMSEMap extends MapReduceBase implements
        Mapper<LongWritable, Text, Text, Text> {

            private Text valText = new Text();
            private Text zero = new Text("0");

            public void map (LongWritable key, Text value,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                String[] toks = value.toString().split("\\s");
                float v = Float.parseFloat(toks[3]);
                if (v > -99.0) {
                    valText.set("1 " + (float)Math.pow(initialVal*initialVal - v, 2));
                    output.collect(zero, valText);
                }
            }
        }

    // a local combiner speeds things up for the reduction. i have a slightly different
    // implementation from the reducer class, the emitted tuple has a String zero
    // other than nulls so that they can be aggregated
    public static class InitRMSECombine extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            private Text valText = new Text();
            private Text zero = new Text("0");

            public void reduce(Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                // sum up the differences to obtain the very first RMSE
                float sum = 0.0F;
                int count = 0;
                while (values.hasNext()) {
                    String[] pair = values.next().toString().split(" ");
                    count += Integer.parseInt(pair[0]);
                    sum += Float.parseFloat(pair[1]);
                }
                valText.set(count + " " + sum);
                output.collect(zero, valText);
            }
        }


    // reducer exactly like combiner: sum up all squared differences and return
    // one tuple with sqrt(sum/count)
    public static class InitRMSEReduce extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            private Text valText = new Text();
            private Text zero = new Text("0");

            public void reduce(Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                // sum up the differences to obtain the very first RMSE
                float sum = 0.0F;
                int count = 0;
                while (values.hasNext()) {
                    String[] pair = values.next().toString().split(" ");
                    count += Integer.parseInt(pair[0]);
                    sum += Float.parseFloat(pair[1]);
                }
                valText.set("" + ((float)Math.sqrt(sum/count)));
                output.collect(null, valText);
            }
        }



    // ------------------------------------------------------------------------------------



    // this function can be called for either U or V elements. it needs
    // one full matrix, and the other matrix only needs to contain required
    // values for that calculation. the M is a map that contains
    // elements in the original matrix also needed for this computation
    public static float calc_new_element(
            float[][] U,
            float[][] V,
            Map<Integer, Float> M,
            int r, int c, boolean u_element) {

        float sum_i = 0.0F, sum_denom = 0.0F;

        // loop over M rather than the whole row or column, this is faster
        for (Map.Entry<Integer, Float> e : M.entrySet()) {
            int i = e.getKey();
            float mVal = e.getValue();

            if (mVal < -20)
                continue;

            float sum_k = 0.0F;
            for (int k = 1; k <= d; ++k) {
                if (u_element && k != c) {
                    sum_k += U[r][k] * V[k][i];
                }
                if (!u_element && k != r) {
                    sum_k += U[i][k] * V[k][c];
                }
            }
            if (u_element) {
                sum_i += V[c][i] * (mVal - sum_k);
                sum_denom += Math.pow(V[c][i], 2);
            }
            else {
                sum_i += U[i][r] * (mVal - sum_k);
                sum_denom += Math.pow(U[i][r], 2);
            }

        }
        return sum_i / sum_denom;
    }

    // to calculate the RMSE of one element
    public static float getMultiplication(float[][] U, float[][] V, int r, int c) {
        float result = 0.0F;
        for (int k = 1; k <= d; ++k)
            result += U[r][k] * V[k][c];
        return result;
    }


    // ------------------------------------------------------------------------------------

    // iteration_U mapper distributes rows to reducers to optimize
    public static class IterMap_U extends MapReduceBase implements
        Mapper<LongWritable, Text, Text, Text> {

            Text keyLong = new Text();

            public void map(LongWritable key, Text value,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                String[] toks = value.toString().split("\\s");
                keyLong.set(toks[1]);
                output.collect(keyLong, value);

            }
        }

    // iteration_U reducer reads entire V matrix, and receives all values
    // for a certain row (user) from the mapper, and its corresponding M vlaues
    // and calculates new values for the row elements
    public static class IterReduce_U extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            float[][] V, U;
            private Text valText = new Text();
            Float mVal;
            int[] cols = new int[d];

            public void configure(JobConf job) {

                // load the V matrix from disk for first time only
                U = new float[nRows + 1][d + 1];
                V = new float[d + 1][nCols + 1];
                for (float[] row : U)
                    Arrays.fill(row, dummyVal);
                String vPath = job.get("vPath");
                buildMatrix(V, vPath);

                for (int i = 1; i <= d; ++i)
                    cols[i - 1] = i;
            }

            public void reduce(Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                Map<Integer, Float> M = new HashMap<Integer, Float>();
                int changedRow = -1;
                while (values.hasNext()) {
                    String[] toks = values.next().toString().split("\\s");
                    int r = Integer.parseInt(toks[1]);
                    int c = Integer.parseInt(toks[2]);

                    if (toks[0].equals("M")) {
                        M.put(c, Float.parseFloat(toks[3]));
                    } else if (toks[0].equals("U")) {
                        U[r][c] = Float.parseFloat(toks[3]);
                        if (changedRow == -1)
                            changedRow = r;
                    }
                }

                // now we calculate new values for U elements
                for (int sub = 0; sub < MAX_SUB_ITER; ++sub) {
                    Collections.shuffle(Arrays.asList(cols));
                    for (int i = 0; i < d; ++i) {
                        float newVal = calc_new_element(U, V, M, changedRow, cols[i], true);
                        if (Float.isNaN(newVal))
                            newVal = 0.0F;
                        U[changedRow][cols[i]] = newVal;
                    }
                }
                for (int i = 0; i < d; ++i) {
                    valText.set("U " + changedRow + " " + cols[i] + " " + U[changedRow][cols[i]]);
                    output.collect(null, valText);
                }

            }
        }


    // ------------------------------------------------------------------------------------


    // iteration_U mapper distributes rows to reducers to optimize
    public static class IterMap_V extends MapReduceBase implements
        Mapper<LongWritable, Text, Text, Text> {

            Text keyLong = new Text();

            public void map(LongWritable key, Text value,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                String[] toks = value.toString().split("\\s");
                keyLong.set(toks[2]);
                output.collect(keyLong, value);

            }
        }

    // iteration_V reducer reads entire U matrix, and receives all values
    // for a certain column (movie) from the mapper, and its corresponding M vlaues
    // and calculates new values for the column elements
    public static class IterReduce_V extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            float[][] U, V;
            private Text valText = new Text();
            Float mVal;
            int[] rows = new int[d];

            public void configure(JobConf job) {

                // load the U matrix from disk for first time only
                U = new float[nRows + 1][d + 1];
                V = new float[d + 1][nCols + 1];
                for (float[] row : V)
                    Arrays.fill(row, dummyVal);
                String uPath = job.get("uPath");
                buildMatrix(U, uPath);

                for (int i = 1; i <= d; ++i)
                    rows[i - 1] = i;
            }

            public void reduce(Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                Map<Integer, Float> M = new HashMap<Integer, Float>();
                int changedCol = -1;
                while (values.hasNext()) {
                    String[] toks = values.next().toString().split("\\s");
                    int r = Integer.parseInt(toks[1]);
                    int c = Integer.parseInt(toks[2]);

                    if (toks[0].equals("M")) {
                        M.put(r, Float.parseFloat(toks[3]));
                    } else if (toks[0].equals("V")) {
                        V[r][c] = Float.parseFloat(toks[3]);
                        if (changedCol == -1)
                            changedCol = c;
                    }
                }

                // now we calculate new values for U elements
                for (int sub = 0; sub < MAX_SUB_ITER; ++sub) {
                    Collections.shuffle(Arrays.asList(rows));
                    for (int i = 0; i < d; ++i) {
                        float newVal = calc_new_element(U, V, M, rows[i], changedCol, false);
                        if (Float.isNaN(newVal))
                            newVal = 0.0F;
                        V[rows[i]][changedCol] = newVal;
                    }
                }
                for (int i = 0; i < d; ++i) {
                    valText.set("V " + rows[i] + " " + changedCol + " " + V[rows[i]][changedCol]);
                    output.collect(null, valText);
                }
            }
        }


    // ------------------------------------------------------------------------------------


    // mapper reads entries from the M matrix, and loads both U and V
    // matrices once, then emit one tuple per calculated P value
    public static class CalcRMSEMap extends MapReduceBase implements
        Mapper<LongWritable, Text, Text, Text> {

            private Text valText = new Text();
            private Text zero = new Text("0");
            private float[][] U, V;

            public void configure(JobConf job) {
                // load the U and V matrices from disk for first time only
                U = new float[nRows + 1][d + 1];
                V = new float[d + 1][nCols + 1];
                String uPath = job.get("uPath");
                String vPath = job.get("vPath");
                buildMatrix(U, uPath);
                buildMatrix(V, vPath);
            }

            public void map(LongWritable key, Text value,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                String[] toks = value.toString().split("\\s");
                int r = Integer.parseInt(toks[1]);
                int c = Integer.parseInt(toks[2]);
                float v = Float.parseFloat(toks[3]);
                valText.set("" + ((float)Math.pow(getMultiplication(U, V, r, c) - v, 2)));
                output.collect(zero, valText);
            }
        }

    // like reducer, a combiner sums up the squared differences and reports
    // the count of the elements
    public static class CalcRMSECombine extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            private Text valText = new Text();
            private Text zero = new Text("0");

            public void reduce (Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                float rmse = 0.0F;
                int count = 0;
                while (values.hasNext()) {
                    rmse += Float.parseFloat(values.next().toString());
                    ++count;
                }

                valText.set(count + " " + rmse);
                output.collect(zero, valText);
            }

        }

    // like the combiner, except that it also divides by the count and
    // returns one value that is the final RMSE
    public static class CalcRMSEReduce extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

            private Text valText = new Text();
            private Text zero = new Text("0");

            public void reduce (Text key, Iterator<Text> values,
                    OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

                float rmse = 0.0F;
                int count = 0;
                while (values.hasNext()) {
                    String[] pair = values.next().toString().split(" ");
                    count += Integer.parseInt(pair[0]);
                    rmse += Float.parseFloat(pair[1]);
                }

                valText.set("" + ((float)Math.sqrt(rmse/count)));
                output.collect(null, valText);
            }
        }

    // ------------------------------------------------------------------------------------
    

    // utility function
    private static String get_time() {
        Date instant = new Date(System.currentTimeMillis());
        SimpleDateFormat simple = new SimpleDateFormat("HH:mm");
        return simple.format(instant);
    }

    public static void calcRMSEJob(int iteration, String outFile) {

        JobConf calcRMSEConf = new JobConf(Main.class);
        calcRMSEConf.setJobName("retrieve the banana - " + iteration);

        calcRMSEConf.setOutputKeyClass(Text.class);
        calcRMSEConf.setOutputValueClass(Text.class);
        calcRMSEConf.setMapperClass(CalcRMSEMap.class);
        calcRMSEConf.setCombinerClass(CalcRMSECombine.class);
        calcRMSEConf.setReducerClass(CalcRMSEReduce.class);

        calcRMSEConf.setNumMapTasks(NUM_REDUCERS);
        calcRMSEConf.setNumReduceTasks(1);

        // send the latest paths for the U and V matrices
        cout("calculating RSME for U@" + uPath + " and V@" + vPath + "\n");
        calcRMSEConf.set("uPath", homeDir + uPath);
        calcRMSEConf.set("vPath", homeDir + vPath);

        FileInputFormat.setInputPaths(calcRMSEConf, new Path(homeDir + "/M"));
        FileOutputFormat.setOutputPath(calcRMSEConf, new Path(homeDir
                    + "/" + outFile + (iteration + 1)));

        calcRMSEConf.set("iteration", String.valueOf(iteration));

        try {
            startTime = System.currentTimeMillis();
            JobClient.runJob(calcRMSEConf);
            endTime = System.currentTimeMillis();

            duration = (endTime - startTime)/1000F;
            System.out.println("\n------------------------------------------------------ CALC RMSE duration: " + duration + " seconds\n");
        }
        catch(Exception e){
            cout("exception while trying to run job: " + e + "\n"
                    + Arrays.toString(e.getStackTrace()) + "\n");
            throw new myException(Arrays.toString(e.getStackTrace()));
        }        

    }

    public static void optimizeMatrixJob(boolean UMatrix, int iteration) {

        JobConf iterConf = new JobConf(Main.class);
        if (UMatrix)
            iterConf.setJobName("KILL MONKEY! - " + iteration + " " + get_time());
        else
            iterConf.setJobName("VROOM VRRROOM! - " + iteration + " " + get_time());

        iterConf.setOutputKeyClass(Text.class);
        iterConf.setOutputValueClass(Text.class);
        iterConf.setInputFormat(TextInputFormat.class);
        iterConf.setOutputFormat(TextOutputFormat.class);

        iterConf.setNumMapTasks(NUM_MAPPERS);
        iterConf.setNumReduceTasks(NUM_REDUCERS);

        if (UMatrix) {
            iterConf.setMapperClass(IterMap_U.class);
            iterConf.setReducerClass(IterReduce_U.class);
        }
        else {
            iterConf.setMapperClass(IterMap_V.class);
            iterConf.setReducerClass(IterReduce_V.class);
        }

        Path[] matricesPaths = new Path[2];
        if (UMatrix)
            matricesPaths[0] = new Path(homeDir + uPath);
        else
            matricesPaths[0] = new Path(homeDir + vPath);
        matricesPaths[1] = new Path(homeDir + "/M");

        iterConf.set("uPath", homeDir + uPath);
        iterConf.set("vPath", homeDir + vPath);

        if (UMatrix)
            uPath = "/U_" + (iteration + 1);
        else
            vPath = "/V_" + (iteration + 1);

        FileInputFormat.setInputPaths(iterConf, matricesPaths);
        if (UMatrix)
            FileOutputFormat.setOutputPath(iterConf, new Path(homeDir + uPath));
        else
            FileOutputFormat.setOutputPath(iterConf, new Path(homeDir + vPath));

        try {

            startTime = System.currentTimeMillis();
            JobClient.runJob(iterConf);
            endTime = System.currentTimeMillis();
            duration = (endTime - startTime)/1000F;
            if (UMatrix)
                System.out.println("\n------------------------------------------------------ U ITERATION duration: " + duration + " seconds\n");
            else
                System.out.println("\n------------------------------------------------------ V ITERATION duration: " + duration + " seconds\n");

        }
        catch(Exception e){
            cout("exception while trying to run job: " + e + "\n"
                    + Arrays.toString(e.getStackTrace()) + "\n");
            throw new myException(Arrays.toString(e.getStackTrace()));
        }

    }

    public static void main(String[] args) throws Exception {

        // set the homeDir
        homeDir = args[1];

        // first job: pre-processing
        // 1- remove the unnecessary date field
        // 2- build and normalize matrix
        JobConf preConf = new JobConf(Main.class);
        preConf.setJobName("## scrrrtch me hed ##\n" + get_time());

        preConf.setOutputKeyClass(LongWritable.class);
        preConf.setOutputValueClass(Text.class);
        preConf.setMapperClass(PreProcessMap.class);
        preConf.setReducerClass(PreProcessReduce.class);
        preConf.setInputFormat(TextInputFormat.class);
        preConf.setOutputFormat(TextOutputFormat.class);
        preConf.setMapOutputKeyClass(Text.class);
        preConf.setMapOutputValueClass(Text.class);

        preConf.setNumMapTasks(NUM_MAPPERS);
        preConf.setNumReduceTasks(NUM_REDUCERS);

        FileInputFormat.setInputPaths(preConf, new Path(args[0]));
        FileOutputFormat.setOutputPath(preConf, new Path(homeDir + "/M"));

        startTime = System.currentTimeMillis();
        JobClient.runJob(preConf);
        endTime = System.currentTimeMillis();

        duration = (endTime - startTime)/1000F;
        System.out.println("\n------------------------------------------------------ PRE-PROCESSING duration: " + duration + " seconds\n");


        // second job: calculate initial RMSE
        JobConf initRMSEConf = new JobConf(Main.class);
        initRMSEConf.setJobName("get yer rmse");

        initRMSEConf.setOutputKeyClass(Text.class);
        initRMSEConf.setOutputValueClass(Text.class);
        initRMSEConf.setMapperClass(InitRMSEMap.class);
        initRMSEConf.setReducerClass(InitRMSEReduce.class);
        initRMSEConf.setCombinerClass(InitRMSECombine.class);
        initRMSEConf.setInputFormat(TextInputFormat.class);
        initRMSEConf.setOutputFormat(TextOutputFormat.class);
        initRMSEConf.setMapOutputValueClass(Text.class);

        initRMSEConf.setNumMapTasks(NUM_TOTAL_WORKERS);
        initRMSEConf.setNumReduceTasks(1);

        FileInputFormat.setInputPaths(initRMSEConf, new Path(homeDir + "/M"));
        FileOutputFormat.setOutputPath(initRMSEConf, new Path(homeDir + "/_rmse0"));

        startTime = System.currentTimeMillis();
        JobClient.runJob(initRMSEConf);
        endTime = System.currentTimeMillis();

        duration = (endTime - startTime)/1000F;
        System.out.println("\n------------------------------------------------------ INITIAL RMSE duration: " + duration + " seconds\n");


        // now create the U, V, and P matrices on disk
        try {
            FileSystem fs = FileSystem.get(new Configuration());
            Path uFilePath = new Path(homeDir + "/U_0/" + uMatrixFile);
            Path vFilePath = new Path(homeDir + "/V_0/" + vMatrixFile);
            BufferedWriter ubr = new BufferedWriter(new OutputStreamWriter(
                        fs.create(uFilePath, true)));
            BufferedWriter vbr = new BufferedWriter(new OutputStreamWriter(
                        fs.create(vFilePath, true)));

            boolean flag = true;
            for (int i = 1; i <= nRows; ++i)
                for (int j = 1; j <= d; ++j) {
                    flag = !flag;
                    ubr.write("U " + i + " " + j + " " + (flag?initialVal:"-"+initialVal) + "\n");
                }

            for (int i = 1; i <= d; ++i)
                for (int j = 1; j <= nCols; ++j) {
                    flag = !flag;
                    vbr.write("V " + i + " " + j + " " + (flag?initialVal:"-"+initialVal)+ "\n");
                }

            ubr.close();
            vbr.close();

        } catch (Exception e) {
            cout("File could not be created\n");
            throw new myException(Arrays.toString(e.getStackTrace()));
        }
        
        for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {

            // first iteration job: optimize U matrix
            cout("running U optimization for iteration " + iteration + "\n");
            optimizeMatrixJob(true, iteration);

            // second iteration job: optimize V matrix
            cout("running V optimization for iteration " + iteration + "\n");
            optimizeMatrixJob(false, iteration);
            
            // calculate new RMSE
            //cout("calculating V RMSE for iteration " + iteration + "\n");
            //calcRMSEJob(iteration, "__rmse");
            
        }

    }
}

