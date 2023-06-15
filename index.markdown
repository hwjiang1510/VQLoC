---

layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>

<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>VQLoC</title>



<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->

<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>
<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>



<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong><br>Single-Stage Visual Query Localization <br /> in Egocentric Videos</strong></h1></center>
<center><h2>
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://srama2512.github.io/">Santhosh Ramakrishnan<sup>1,2</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.cs.utexas.edu/users/grauman/">Kristen Grauman<sup>1,2</sup></a>&nbsp;&nbsp;&nbsp; 
   </h2>
    <center><h3>
        <a href="https://www.cs.utexas.edu/"><sup>1</sup>UT Austin</a>&nbsp;&nbsp;&nbsp;
        <a href="https://ai.facebook.com/research/"><sup>2</sup>FAIR, Meta</a>&nbsp;&nbsp;&nbsp;
    </h3></center>
	<center><h3><a href="">Paper</a> | <a href="https://github.com/hwjiang1510/VQLoC">Code</a> </h3></center>





<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Visual Query Localization on long-form egocentric videos requires spatio-temporal search and localization of visually specified objects and is vital to build episodic memory systems. Prior work develops complex multi-stage pipelines that leverage well-established object detection and tracking methods to perform VQL. However, each stage is independently trained and the complexity of the pipeline results in slow inference speeds. We propose VQLoC, a novel single-stage VQL framework that is end-to-end trainable. Our key idea is to first build a holistic understanding of the query-video relationship and then perform spatio-temporal localization in a single shot manner. Specifically, we establish the query-video relationship by jointly considering query-to-frame correspondences between the query and each video frame and frame-to-frame correspondences between nearby video frames. Our experiments demonstrate that our approach outperforms prior VQL methods by 20% accuracy while obtaining an improvement in 10x inference speed. VQLoC is also the top entry on the Ego4D VQ2D challenge leaderboard.
</p></td></tr></table>
</p>
  </div>
</p>



<h1 align="center">Problem Definition</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay width="100%">
        <source src="./video/task.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
      The goal is to search and localize open-set visual queries in long-form videos, jointly predicting the appearance time window and the corresponding spatial bounding boxes.
</p></td></tr></table>

<br>

<h1 align="center">Challenges</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay loop width="70%">
        <source src="./video/challenge.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
      The challenges of the task arise from: i) the "needle-in-the-haystack" problem; ii) the diversity of open-set queries, and iii) no "exact" matching in the target videos.
</p></td></tr></table>

<br>

<hr> <h1 align="center">Method</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/pipeline.png"> <img
src="./src/pipeline.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>
<table width=800px><tr><td> <p align="justify" width="20%">Our model VQLoC estabishing a holistic undertsanding of query-video relationship, and predicts the results based on the understanding in a single shot. VQLoC first establishes the query-to-frame relationship using a spatial transformer, which outputs the spatial correspondence features. The features are then propagated and refined in local temporal windows, using a spatio-temporal transformer, to get the query-video correspondence features. Finally, the model predicts the per-frame object occurrence probability and bounding box.</p></td></tr></table>
<br>
<hr> <h1 align="center">Results</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/tradeoff.png"> <img
src="./src/tradeoff.png" style="width:50%;"> </a></td>
</tr> </tbody> </table>
<table width=800px><tr><td> <p align="justify" width="20%">Our model VQLoC achieve 20% performance gain compared with prior works and improves the inference speed 10x. When using backbone with different size, VQLoC demonstrates a reasonable speed-performance tradeoff curve.</p></td></tr></table>
<br>


<hr>
<h1 align="center">Visualization</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay loop width="70%">
        <source src="./video/vis.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>


  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
      We show identified query object response track with bounding boxes.
</p></td></tr></table>

<br>



<hr>
<h1 align="center">Video</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted width="70%">
        <source src="./video/video_full.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>


<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>
<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@article{jiang2023vqloc,
   title={Single-Stage Visual Query Localization in Egocentric Videos},
   author={Jiang, Hanwen and Ramakrishnan, Santhosh and Grauman, Kristen},
   journal={ArXiv},
   year={2023},
   volume={}
}
</code></pre>
</left></td></tr></table>




<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>

<center><h1>Acknowledgements</h1></center> 
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

