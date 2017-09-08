# FSbasedMultiLabelStreamClassification
Max-Relevance and Min-Redundancy based Multi-label Data Stream Classification with Concept Drifting Detection
<P>Multi-label data stream classification is a very challeng-
ing and significant task especially in the handling of high-
dimensional data streams with concept drifts. However, this
challenge has received little attention from the research com-
munity. Therefore, we propose a max-relevance and min-
redundancy based algorithm adaptation approach for the effi-
cient and effective classification on multi-label data streams
with high-dimensional attributes and concept drifts1. In or-
der to reduce the impact from high-dimensional attributes
with noisy attributes, we first refine the minimal-redundancy-
maximal-relevance criterion based on mutual information
to select qualified features. Secondly, we propose the label
and feature distribution based concept drifting detection ap-
proach to distinguish concept drifts hidden in multi-label data
streams. Finally, we build an incremental ensemble classifica-
tion model for efficiently classifying multi-label data streams.
Extensive studies show that our approach can get optimal sub-
sets of features while maintaining a good performance in the
multi-label classification, as compared to several state-of-the-
art multi-label feature selection algorithms using two efficient
multi-label classification methods as base classifiers.</P>

<H2>Introduction</H2>
<P>Multi-label classification is a challenging and popular topic
in many real-world applications, such as text categorization,
image annotation and gene function classification. These da-
ta present a typical character that one instance is possibly as-
sociated with multiple labels simultaneously. For example, a
piece of micro blog concerning haze may belong to the cat-
egories of environment protection, weather and advertise-
ment. It is hence a challenge for the traditional single-label
classification with the assumption that each data point be-
longs to exactly one label. In recent years, these multi-label
data swept theWeb at an alarming rate, and have produced a
large quantity of data stream, called multi-label data stream-
s. Multi-label data stream classification is hence a challeng-
ing task owing to the following facts.</P>
<P>First, the data are of explosive growth and high dimen-
sionality with irrelevant and redundant features. This is es-
pecially true for Web texts and images. For example, Web access data 
from Yahoo website are up to 195.16 million times in December, 2013. 
Filckr, a public picture sharing website receives 35 million photos per day on average till November, 2013. 
These data usually contain thousands or even tens of thousands of features. Second, they
present another obvious characteristic of data streams as being concept drifting (Gama et al. 2014), namely the data
distributions of class labels or values of several attributes change over time. For example, interests of users probably
change from the hot news to popular shopping products over time. However, traditional multi-label data classification approaches (Tai et al. 2012; Liu et al. 2015; Jian et al. 2016;
Yeh et al. 2017) rarely focus on the handling of all above
data stream characteristics. Correspondingly, some multi-label data stream classification approaches (Read et al. 2012;
Osojnik, Panov, and D\^{z}eroski 2017) have been proposed, but
they all miss the issue of high dimensionality in multi-label
data stream while some of them miss the concept drifts.</P>
<P>The high dimensionality of multi-label data significantly
increases the computational burden for many learning algorithms, but also limits the usage of these algorithms in real-
world applications due to the curse of dimensionality (Duda,
Hart, and Stork 2012). To overcome this problem, many dimensionality reduction based multi-label learning approach-
es including feature extraction (FE) (Tai et al. 2012; Liu et al.
2015; Zhang and Zhou 2008; Sun, Ji, and Ye 2011) and feature selection (FS) (Jian et al. 2016; Gu, Li, and Han 2011;
Doquire and Verleysen 2013; Pandey and Vaze 2014) have
been proposed. Multi-label FE techniques focus on transforming the original feature space to a new space, which
usually impose more computational burden as compared to
multi-label FS techniques. Because the latter aims to select a small subset of features that minimizes feature-feature
redundancy and maximizes feature-label relevance simultaneously, in which it preserves the physical meanings of the
original data. In this paper, we focus on multi-label Feature
Selection (FS) techniques for efficiency.</P>
<P>Existing multi-label FS approaches mainly follow two directions (Gibaja and Ventura 2015; Pereira et al. 2016). One
intuitive approach is Problem Transformation (PT) based FS approach (Spola\^{o}r et al. 2013), which transforms a multi-
label problem into one or more single label problems, and
then evaluates each single label using traditional feature selection techniques, including Fisher score, Chi-Square, Information Gain (Spola\^{o}r et al. 2013), 
ReliefF (Reyes, CarlosMorell, and Ventura 2015), to name a few. However, this
approach will result in subsequent problems especially in the
handling of large-scale multi-label data streams. First, each
label is treated independently, which misses the label corre-
lations. Second, the newly created label contains too many
classes, leading to learning difficulties due to the effective-
ness and the efficiency.</P>
<P>The extension-type approach is the Algorithm Adaptation (AA) based FS approach, in which the algorithm is in-
ternally adapted to multi-label problems. We can divide it
into three categories by the partition for single-label feature selection, including wrapper, embedded and filter (Xu 2016). More specifically, Wrapper approaches iteratively
apply a search strategy to search the space of all possible
feature subsets and evaluate their corresponding classification performance using some classifier, such as SVM (Gu,
Li, and Han 2011), Random Forest (Spola\^{o}r et al. 2013), kNN (Zhang and Zhou 2007), and genetic-algorithm (Lee and
Kim 2015b). Embedded approaches directly incorporate FS
as a part of the learner training process, such as the mutual
information based technique for informative label selection
(Pandey and Vaze 2014), and the \ell 1-norm regularization
based multi-label informed FS (Jian et al. 2016). Such two
kinds of approaches consider the correlation among labels, but both have an extremely high time complexity due to
requiring training a large number of classifiers or their complicated optimization procedures (Aggarwal 2015).</P>


<H2>Our Approach</H2>
<P>Contrary to the above approaches, filter approaches are
independent of any classification algorithm, and they usually evaluate the usefulness of a feature, or a set of fea-
tures, through measures of distance (Reyes, CarlosMorell,
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
<P>Our main contributions of this paper are as follows</P>
<P>First, our approach can produce the higher accuracy of
feature selection. In terms of advantages of the AA multilabel learning approach and the filter approach, we still aim
at designing and implementing a novel extension-type filter FS approach for multi-label data stream classification.
Unlike existing multi-label filter FS approaches (Lin et al.
2016), we use a sliding window to build an ensemble model incrementally for adapting to multi-label data streams, and then we give the analysis of generalization error of
the ensemble model. Meanwhile, we extend the minimalredundancy-maximal-relevance criterion based on mutual information for single-label classification (Peng, Long, and
Ding 2005) to multi-label data classification. This is because
mutual information is a submodular function, which can provide a theoretical guarantee on the quality of a subset select-
ed in the feature selection.</P>

<P>Second, our approach can detect concept drifts hidden in
multi-label data streams. To track concept drifts hidden in
multi-label data streams, we propose a concept drifting de-
tection method based on the label distribution and the fea-
ture distribution. It is capable of capturing concept drift-
s in multi-label data streams effectively. Contrary to the
classification-error based concept drifting detection method
in the data stream classification such as (Gama et al. 2014;
Frias-Blanco et al. 2015), we define the difference of data
distributions between two adjoining data chunks, and then
detect whether concept drifts occur due to the changing of
the label distribution or the feature distribution.</P>

<P>Finally, our approach can perform efficiently in the han-
dling of multi-label data streams. The model used here is
incremental, the time cost is relevant to the size of a data
chunk, while the time costs in aforementioned multi-label
FS approaches depend on the size of the whole multi-label
data set or the square value. Thus, our approach is more ef-
ficient and scalable.</P>
<H2>Data Set </H2>
<P><A onclick="stc(this, 26)" href="http://mulan.sourceforge.net/datasets-mlc.html" 
target="_new"> Benchark data sets</A>: In our experiments, we select six large-
scale benchmark multi-label databases from different appli-
cation domains to simulate the multi-label data stream. De-
tails of these data sets are listed in Table 1, where Label-
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
<P>Our approach is implemented in an open-source Java library for learning from Multi-label data, called Mulan <A onclick="stc(this, 26)" href="https://github.com/BlindReviewAAAI18/FSbasedMultiLabelStreamClassification/blob/master/ML_MRMR_FSClassification.java" 
target="_new">. Source codes of our approach include the feature selection for multi-label data stream in the <A onclick="stc(this, 26)" href="https://github.com/BlindReviewAAAI18/FSbasedMultiLabelStreamClassification/blob/master/feasel.rar" 
target="_new"> feasel zip file </A>, and the ML_MRMR_FSClassification java file.</P>
</DIV>
<H2>Parameter Discription</H2>
<P> 
<TABLE width="700" align="center" class=" borderColumns borderRows tableBorder" 
cellSpacing="0" cellPadding="0">
  <TBODY>
  <TR>
    <TD align="center"><B>/******Parameter Discription***********/</B></TD>
	<TD align="center"><B>"-alph": the threshold used in the selection on an optimal subset in MRMR based feature selection, default alph = 0.2;</B></TD>
	<TD align="center"><B>"-blta": the threshold used in the drifting detection based on the class distribution: default blta = 0.2;</B></TD>
	<TD align="center"><B>"-gamma": the threshold used in the drifting detection based on the feature distribution: default gamma = 0.2;</B></TD>
	<TD align="center"><B>"-dataBlock": the size of a data chunk, default dataBlock = 200;</B></TD>
	<TD align="center"><B>"-modelSize": the number of models in the ensemble model, default modelSize = 100;</B></TD>

	<TD align="center"><B>"-path": the file directory;</B></TD>
	<TD align="center"><B>"-arff": the source file;</B></TD>
	<TD align="center"><B>"-test": the testing file: </B></TD>
	<TD align="center"><B>"-attrSize": the size of feature space, namely the attribute count+the label count;</B></TD>
	<TD align="center"><B>"-labelNum": the label count;</B></TD>
	<TD align="center"><B>"-simElvType": the type of similarity evaluation, default value "Jaccard";</B></TD>
	<TD align="center"><B>"-algType": the type of algorithms, default value "MLKNN";----it is useless if you select MLRDT as a base classifier;</B></TD>
	<TD align="center"><B>"-bDiscretized": the flag of discretization, default "false";</B></TD>
	<TD align="center"><B>"-bAvgVoting": the flag of voting, default "true";</B></TD>
  </TR>
</TBODY></TABLE>
</P>
<H2>Demo code "how to install our approach"</H2>
<P> Please decompress the <B>feasel</B> zip file, and put this folder under the directory of "src" folder at mulan project. In our project, we put the file ML_MRMR_FSClassification.java in the folder 
of "/src/mulan/examples", and it has the main function. You can use the following demos to run our approach.
<P>Demo: how to install our approach using MLRDT as the base classifier, in this case, we select the Corel16k010 data set as a demo data set;

```Java
public static void main(String[] args) throws Exception {
		/*********Classify by MLRDT after ML-MRMR-Feature selection**************/
		String[] comParms = {"-alph", "0.2", "-blta", "0.2", "-gamma", "0.2", "-dataBlock", "200", "-modelSize", "100"};
		ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
		mcf.InitComParms(comParms);
		String[] options = {"-path","H:/data/Corel16k010","-train","Corel16k010-train.arff-sort.arff","-test", "Corel16k010-test.arff","-xml","Corel16k010.xml",
			"-attrSize","644", "-labelNum","144", "-minS", "4", "-treeNum", "10", "-simElvType", "Jaccard", "-bDiscretized", "false", "-bAvgVoting", "true"};
		mcf.ML_MRMR_FS_ClassifyByMLRDT(options);
}</P>
<P>
Demo: how to install our approach using MLKNN as the base classifier

```Java
public static void main(String[] args) throws Exception {
	  /****************Classify by mulan after ML-MRMR-Feature selection**********************/
	  String[] comParms = {"-alph", "0.2", "-blta", "0.2", "-gamma", "0.2", "-dataBlock", "200", "-modelSize", "100"};
	  ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
		mcf.InitComParms(comParms);
      String[] options = {"-path","H:/data/Corel16k010","-train","Corel16k010-train.arff-sort.arff","-test", "Corel16k010-test.arff","-xml","Corel16k010.xml",
			"-attrSize","644", "-labelNum","144", "-simElvType", "Jaccard", "-algType", "MLKNN", "-bDiscretized", "false","-bAvgVoting", "true"};
		ML_MRMR_FSClassification mcf = new ML_MRMR_FSClassification();
		mcf.ML_MRMR_FS_ClassifyByMulan(options);
	}
</P>
       
<DIV style="clear: both;"></DIV>
<DIV class="conM ">
<H2>Used Data Sets: Download</H2>
<P>More Details Refer to <A onclick="stc(this, 26)" href="http://121.42.218.45/peipeili/Used-Data-Sets-and-demo.rar" 
target="_new"> Used Data Sets</A>.</P></DIV>
<DIV style="clear: both;"></DIV>
<DIV class="conM ">
<H2>Source codes: Download</H2>
<P>More Details Refer to <A onclick="stc(this, 26)" href="http://121.42.218.45/peipeili/ShortTextClassification-src.rar" 
target="_new"> Source codes</A>.</P></DIV>
