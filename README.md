# FSbasedMultiLabelStreamClassification
<B>Max-Relevance and Min-Redundancy based Multi-label Data Stream Classification with Concept Drifting Detection</B>
<P>Multi-label data stream classification is a very challenging and significant task especially in the handling of high-dimensional data streams with concept drifts. However, this
challenge has received little attention from the research community. Therefore, we propose a max-relevance and min-redundancy based algorithm adaptation approach for the efficient and effective classification on multi-label data streams
with high-dimensional attributes and concept drifts1. In order to reduce the impact from high-dimensional attributes
with noisy attributes, we first refine the minimal-redundancy-maximal-relevance criterion based on mutual information
to select qualified features. Secondly, we propose the label
and feature distribution based concept drifting detection approach to distinguish concept drifts hidden in multi-label data
streams. Finally, we build an incremental ensemble classification model for efficiently classifying multi-label data streams.
Extensive studies show that our approach can get optimal subsets of features while maintaining a good performance in the
multi-label classification, as compared to several state-of-the-art multi-label feature selection algorithms using two efficient
multi-label classification methods as base classifiers.</P>

<H2>Our Approach</H2>
<P>Contrary to the above approaches, filter approaches are
independent of any classification algorithm, and they usually evaluate the usefulness of a feature, or a set of features, through measures of distance (Reyes, CarlosMorell,
and Ventura 2015), dependency, information or correlation
on data (Lin et al. 2016). Thus, the biases of learning algorithms do not influence feature selection, and meanwhile
they have the advantage of being fast and simple to implement. However, all aforementioned approaches are batch
ones, and they mainly focus on improving the multi-label
learning accuracy. Thus, they are unsuitable for handling
multi-label data stream directly due to the lower efficiency, not to mention the handling of the hidden concept drifts.
Therefore, in this paper we aim to design an efficient and effective classification approach based on feature selection for
multi-label data stream with concept drifts. To the best of
our knowledge, this is the first feature selection based classification approach for multi-label data streams with high
dimensional features and concept drifts. </P>
<P><B>Our main contributions of this paper are as follows.</B></P>
<P>First, our approach can produce the higher accuracy of feature selection. In terms of advantages of the AA multilabel learning approach and the filter approach, we still aim
at designing and implementing a novel extension-type filter FS approach for multi-label data stream classification.
Unlike existing multi-label filter FS approaches (Lin et al.
2016), we use a sliding window to build an ensemble model incrementally for adapting to multi-label data streams, and then we give the analysis of generalization error of
the ensemble model. Meanwhile, we extend the minimalredundancy-maximal-relevance criterion based on mutual information for single-label classification (Peng, Long, and
Ding 2005) to multi-label data classification. This is because
mutual information is a submodular function, which can provide a theoretical guarantee on the quality of a subset select-
ed in the feature selection.</P>

<P>Second, our approach can detect concept drifts hidden in
multi-label data streams. To track concept drifts hidden in
multi-label data streams, we propose a concept drifting detection method based on the label distribution and the feature distribution. It is capable of capturing concept drifts in multi-label data streams effectively. Contrary to the
classification-error based concept drifting detection method
in the data stream classification such as (Gama et al. 2014;
Frias-Blanco et al. 2015), we define the difference of data
distributions between two adjoining data chunks, and then
detect whether concept drifts occur due to the changing of
the label distribution or the feature distribution.</P>

<P>Finally, our approach can perform efficiently in the handling of multi-label data streams. The model used here is
incremental, the time cost is relevant to the size of a data
chunk, while the time costs in aforementioned multi-label
FS approaches depend on the size of the whole multi-label
data set or the square value. Thus, our approach is more efficient and scalable.</P>
<H2>Data Set </H2>
<P><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new"> Benchark data sets</A>: In our experiments, we select six large scale benchmark multi-label databases from different application domains to simulate the multi-label data stream. Details of these data sets are listed in Table 1, where Label-
Cardinality is the average number of labels in a database
while Label-Density is the average number of labels in a
database divided by the label count L.</P>

<H2>Experiment Results</H2>
<P>Table 1 shows the benchmark data sets used in our experiments, you can download from the file list. 
<!--Due to the space limit, other experimental results are not shown here, you can get details from Download.-->
</P>
<P align="center"><B>Table 1: DATA SETS USED IN THE EXPERIMENTS</B></P>
<P>
<TABLE width="700" align="center" class=" borderColumns borderRows tableBorder" 
cellSpacing="0" cellPadding="0">
  <TBODY>
  <TR>
   <th rowspan="2">Dataset</th>
    <th rowspan="2">domain</th>
    <th colspan="2">Instances</th>
    <th colspan="2">Attributes</th>
    <th rowspan="2"> Labels</th>
    <th rowspan="2"> Label Cardinality</th>
    <th rowspan="2">Laebl Density</th>
  </TR>
  <TR>
    <TD align="center"><B>train</B></TD>
    <TD align="center"><B>test</B></TD>
    <TD align="center"><B>discrete</B></TD>
    <TD align="center"><B>numerical</B></TD>
  </TR>
   <TR>
    <TD align="center"><B><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new">Mediamill</B></TD>
    <TD align="center"><B>vedio</B></TD>
    <TD align="center"><B>30993</B></TD>
    <TD align="center"><B>12914</B></TD>
    <TD align="center"><B>0</B></TD>
    <TD align="center"><B>120</B></TD>
    <TD align="center"><B>101</B></TD>
    <TD align="center"><B>4.376</B></TD>
    <TD align="center"><B>0.043</B></TD>
  </TR>
  <TR>
    <TD align="center"><B><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new">IMDB-ECC-F</B></TD>
    <TD align="center"><B>Movie</B></TD>
    <TD align="center"><B>76143</B></TD>
    <TD align="center"><B>19281</B></TD>
    <TD align="center"><B>1001</B></TD>
    <TD align="center"><B>0</B></TD>
    <TD align="center"><B>28</B></TD>
    <TD align="center"><B>1.920</B></TD>
    <TD align="center"><B>0.036</B></TD>
  </TR>
  <TR>
    <TD align="center"><B><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new">Corel16k010</B></TD>
    <TD align="center"><B>images</B></TD>
    <TD align="center"><B>13618</B></TD>
    <TD align="center"><B>6660</B></TD>
    <TD align="center"><B>500</B></TD>
    <TD align="center"><B>0</B></TD>
    <TD align="center"><B>144</B></TD>
    <TD align="center"><B>2.834</B></TD>
    <TD align="center"><B>0.017</B></TD>
  </TR>
   <TR>
    <TD align="center"><B><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new">NUS-WIDE</B></TD>
    <TD align="center"><B>images</B></TD>
    <TD align="center"><B>161789</B></TD>
    <TD align="center"><B>107859</B></TD>
    <TD align="center"><B>0</B></TD>
    <TD align="center"><B>500</B></TD>
    <TD align="center"><B>81</B></TD>
    <TD align="center"><B>1.869</B></TD>
    <TD align="center"><B>0.023</B></TD>
  </TR>
  <TR>
    <TD align="center"><B><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new">EUR-Lex(subject matters)</B></TD>
    <TD align="center"><B>text</B></TD>
    <TD align="center"><B>17414</B></TD>
    <TD align="center"><B>1935</B></TD>
    <TD align="center"><B>0</B></TD>
    <TD align="center"><B>5000</B></TD>
    <TD align="center"><B>412</B></TD>
    <TD align="center"><B>2.213</B></TD>
    <TD align="center"><B>0.011</B></TD>
  </TR>
  <TR>
    <TD align="center"><B><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new">bookmarks</B></TD>
    <TD align="center"><B>text</B></TD>
    <TD align="center"><B>70045</B></TD>
    <TD align="center"><B>17811</B></TD>
    <TD align="center"><B>2150</B></TD>
    <TD align="center"><B>0</B></TD>
    <TD align="center"><B>208</B></TD>
    <TD align="center"><B>2.028</B></TD>
    <TD align="center"><B>0.010</B></TD>
  </TR>
</TBODY></TABLE>
<P></P>
       
<DIV style="clear: both;"></DIV>
<DIV class="conM ">
<DIV style="clear: both;"></DIV>
<DIV class="conM ">
<H2>Source codes: Download</H2>
<P>Our approach is implemented in an open-source Java library for learning from Multi-label data, called  <A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/index.html" 
target="_new">Mulan </A>.</P>
<P> Source codes of our approach include the feature selection for multi-label data stream in the <A onclick="stc(this, 26)" href="https://github.com/BlindReviewAAAI18/FSbasedMultiLabelStreamClassification/blob/master/feasel.rar" 
target="_new"> feasel</A> zip file, and the <A onclick="stc(this, 26)" href="https://github.com/BlindReviewAAAI18/FSbasedMultiLabelStreamClassification/blob/master/ML_MRMR_FSClassification.java" 
target="_new"> ML_MRMR_FSClassification</A>  java file.</P>
</DIV>
<H2>Parameter Discription</H2>
<P> 
<TABLE width="700" align="left" class=" borderColumns borderRows tableBorder" 
cellSpacing="0" cellPadding="0">
  <TBODY>
	 <TR><TD align="left"><B>"-alph": the threshold used in the selection on an optimal subset in MRMR based feature selection, default alph = 0.2;</B></TD></TR>
	 <TR><TD align="left"><B>"-blta": the threshold used in the drifting detection based on the class distribution: default blta = 0.2;</B></TD></TR>
	 <TR><TD align="left"><B>"-gamma": the threshold used in the drifting detection based on the feature distribution: default gamma = 0.2;</B></TD></TR>
	 <TR><TD align="left"><B>"-dataBlock": the size of a data chunk, default dataBlock = 200;</B></TD></TR>
	 <TR><TD align="left"><B>"-modelSize": the number of models in the ensemble model, default modelSize = 100;</B></TD></TR>
	 <TR><TD align="left"><B>"-path": the file directory;</B></TD></TR>
	 <TR><TD align="left"><B>"-arff": the source file;</B></TD></TR>
	 <TR><TD align="left"><B>"-test": the testing file: </B></TD></TR>
	 <TR><TD align="left"><B>"-attrSize": the size of feature space, namely the attribute count+the label count;</B></TD></TR>
	 <TR><TD align="left"><B>"-labelNum": the label count;</B></TD></TR>
	 <TR><TD align="left"><B>"-simElvType": the type of similarity evaluation, default value "Jaccard";</B></TD></TR>
	 <TR><TD align="left"><B>"-algType": the type of algorithms, default value "MLKNN";----it is useless if you select MLRDT as a base classifier;</B></TD></TR>
	 <TR><TD align="left"><B>"-bDiscretized": the flag of discretization, default "false";</B></TD></TR>
	 <TR><TD align="left"><B>"-bAvgVoting": the flag of voting, default "true";</B></TD></TR>
 </TBODY></TABLE>
</P>
<DIV style="clear: both;"></DIV>
<DIV class="conM ">
<H2>How to install our approach</H2>
<P> Please decompress the <B>feasel</B> zip file, and put this folder under the directory of "src" folder at mulan project. In our project, we put the file ML_MRMR_FSClassification.java in the folder 
of "/src/mulan/examples", and it has the main function. You can use the following demos to run our approach.
<P>Demo: how to install our approach using MLRDT as the base classifier, in this case, we select the Corel16k010 data set as a demo data set;

```Java
public static void main(String[] args) throws Exception {
 *******Classify by MLRDT after ML-MRMR-Feature selection**************/
 String[] comParms = {"-alph", "0.2", "-blta", "0.2", "-gamma", "0.2", "-dataBlock", "200", "-modelSize", "100"};
 ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
 mcf.InitComParms(comParms);
 String[] options = {"-path","H:/data/Corel16k010","-train","Corel16k010-train.arff-sort.arff","-test", "Corel16k010-test.arff","- xml","Corel16k010.xml","-attrSize","644", "-labelNum","144", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized",  "false", "-bAvgVoting", "true"};
 mcf.ML_MRMR_FS_ClassifyByMLRDT(options);
}

Demo: how to install our approach using MLKNN as the base classifier;

public static void main(String[] args) throws Exception {
  /****************Classify by mulan after ML-MRMR-Feature selection**********************/
  String[] comParms = {"-alph", "0.2", "-blta", "0.2", "-gamma", "0.2", "-dataBlock", "200", "-modelSize", "100"};
  ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
  mcf.InitComParms(comParms);
  String[] options = {"-path","H:/data/Corel16k010","-train","Corel16k010-train.arff-sort.arff","-test", "Corel16k010-test.arff","-xml","Corel16k010.xml","-attrSize","644", "-labelNum","144", "-simElvType", "Jaccard", "-algType", "MLKNN", "-bDiscretized", "false","-bAvgVoting", "true"};
ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
mcf.ML_MRMR_FS_ClassifyByMulan(options);
}
