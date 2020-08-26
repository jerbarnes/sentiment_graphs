#!/usr/bin/perl

# Author: Roser Morante
# Institution: CLiPS - University of Antwerp
# Version 2.2
# Date: 1April2012


#use strict;
#use warnings;
use Getopt::Std;



########################################################
########### declaration of variables ###################
########################################################

our ($opt_g, $opt_s, $opt_h, $opt_r) ;


my $line_number = 0;
my $tmp_lineg = "";
my $tmp_linep = "";
my @tmp_lineg = ();
my @tmp_linep = ();

my @POS = ();

my @neg_cols_g = ();
my @neg_cols_p = ();

my @neg_words_g = ();
my @scope_words_g = ();
my @scope_words_nopunc_g = ();
my @negated_words_g = ();

my @neg_tokens_g = ();
my @neg_found_g = ();
my @scope_tokens_g = ();
my @scope_tokens_nopunc_g = ();
my @negated_tokens_g = ();

my @neg_words_p = ();
my @scope_words_p = ();
my @scope_words_nopunc_p = ();
my @negated_words_p = ();

my @neg_tokens_p = ();
my @neg_found_p = ();
my @scope_tokens_p = ();
my @scope_tokens_nopunc_p = ();
my @negated_tokens_p = ();

my @first_token_negs_g = ();
my @first_token_negs_p = ();

my @first_token_negated_g = ();
my @first_token_negated_p = ();

my @first_token_scope_g = ();
my @first_token_scope_p = ();

my @negated_found_g = ();
my @negated_found_p = ();

my @scope_found_g = ();
my @scope_found_p = ();

my $zero_negs_p = "";
my $zero_negs_g = "";

my $fp_cue = 0;
my $fp_scope = 0;
my $fp_scope_nopunc = 0;
my $fp_scope_tokens = 0;
my $fp_negated = 0;

my $fn_cue = 0;
my $fn_scope = 0;
my $fn_scope_nopunc = 0;
my $fn_scope_tokens = 0;
my $fn_negated = 0;

my $tp_cue = 0;
my $tp_scope = 0;
my $tp_scope_nopunc = 0;
my $tp_scope_tokens = 0;
my $tp_negated = 0;

my $fp_full_negation = 0;
my $fn_full_negation = 0;
my $tp_full_negation = 0;

my $fp_negated_apart = 0;
my $tp_negated_apart = 0;
my $fn_negated_apart = 0;

my $fp_scope_apart = 0;
my $tp_scope_apart = 0;
my $fn_scope_apart = 0;

my $cues_g = 0;
my $scopes_g = 0;
my $scope_tokens_g = 0;
my $negated_g = 0;

my $cues_p = 0;
my $scopes_p = 0;
my $scope_tokens_p = 0;
my $negated_p = 0;

my $total_scope_tokens_g = 0;
my $total_scope_tokens_p = 0;


my $error_found = 0;

my $count_sentences = 0;
my $count_sentences_negation = 0;
my $count_error_sentences = 0;
my $count_error_sentences_negation = 0;

my $i = 0;
my $z = 0;
my $x = 0;
my $y = 0;

my $val = 0;
my $found = 0;
my $found_scope_token = 0;


my $st_neg_words_g = "";
my $st_scope_words_g = "";
my $st_scope_words_nopunc_g = "";
my $st_negated_words_g = "";

my $st_neg_tokens_g = "";
my $st_scope_tokens_g = "";
my $st_scope_tokens_nopunc_g = "";
my $st_negated_tokens_g = "";

my $st_neg_words_p = "";
my $st_scope_words_p = "";
my $st_scope_words_nopunc_p = "";
my $st_negated_words_p = "";

my $st_neg_tokens_p = "";
my $st_scope_tokens_p = "";
my $st_scope_tokens_nopunc_p = "";
my $st_negated_tokens_p = "";

my $max_negs_g = -1;
my $max_negs_p = -1;

my $max_negated_g = -1; 
my $max_negated_p = -1; 

my $max_neg_cols_g = -1;
my $max_neg_cols_p = -1;

my $max_lineg = -1;
my $max_linep = -1;

my $max_neg_tokens_p = -1;
my $max_neg_tokens_g = -1;

my $max_scope_tokens_p = -1;
my $max_scope_tokens_g = -1;

my $max_negated_tokens_p = -1;
my $max_negated_tokens_g = -1;
my $max_negated_words_p = -1;
my $max_negated_words_g = -1;

my $max_negated_tokens_nopunc_p = -1;
my $max_negated_tokens_nopunc_g = -1;
my $max_negated_words_nopunc_p = -1;
my $max_negated_words_nopunc_g = -1;
  
  
my $tmp_neg_g = "";
my $tmp_neg_p = "";

my $tmp_word = "";

######################################################################

getopts("g:s:hr") ;

if ((defined $opt_r) || ((! defined $opt_g) && (! defined $opt_s)  && (! defined $opt_h)))
 {
print<<'MYEOT';

 This evaluation script compares the output of a system versys a gold annotation and provides the following information:

 ----------------------------+------+--------+------+------+------+---------------+------------+---------
                             | gold | system | tp   | fp   | fn   | precision (%) | recall (%) | F1  (%) 
 ----------------------------+------+--------+------+------+------+---------------+------------+---------
 Cues:                              |        |      |      |      |               |            |        
 Scopes(cue match):                 |        |      |      |      |               |            |        
 Scopes(no cue match):              |        |      |      |      |               |            |        
 Scope tokens(no cue match):        |        |      |      |      |               |            |        
 Negated(no cue match):             |        |      |      |      |               |            |        
 Full negation:                     |        |      |      |      |               |            |        
 ----------------------------+------+--------+------+------+------+---------------+------------+---------
 Cues B:                            |        |      |      |      |               |            |        
 Scopes B (cue match):              |        |      |      |      |               |            |        
 Scopes B (no cue match):           |        |      |      |      |               |            |        
 Negated B (no cue match):          |        |      |      |      |               |            |        
 Full negation B:                   |        |      |      |      |               |            |         
 ----------------------------+------+--------+------+------+------+---------------+------------+---------
 # sentences:  
 # negation sentences:  
 # negation sentences with errors: 
 % correct sentences:  
 % correct negation sentences:  
 --------------------------------------------------------------------------------------------------------


 The F measures for "cues", "scope", and "negated" are calculated at scope level.
 The F measures for "scope tokens" are calculated at token level counting as tokens the total number of scope tokens. If a sentence has 2 scopes, one with 5 tokens and another with 4, the total number of scope tokens is 9, 

 precision = tp / (tp + fp)
 recall = tp / (tp + fn)
 F1 = (2 * $precision_cue * $recall_cue) / ($precision_cue + $recall_cue)

 For cue, scope and negated to be correct, both, the tokens and the words or parts of words have to be
 correclty identified.

 From v.2.1  the final periods in abbreviations are disregarded. If gold has value "Mr." and system "Mr", system is counted as correct.

 From v.2.2 punctuation tokens are *not* taken into account for evaluation. 

 From v2.2 the "B" variant for Cues, Scopes, Negated and Full Negation has been introduced. 
 The difference lies in how precision is calculated.
 In the "B" measures, precision = tp / system.

 - In "scopes(cue match)", for the scope to be correct, the cue has to be correct.
 - In "scope(no cue match)", for the scope to be correct, the cue doesn't need to be completely correct, though there must be a token
     of overlap between the cue predicted by the system and the gold cue.
 - In "scope tokens(no cue match)", for the scope tokens to be correct, the cue doesn't need to be completely correct, though there must be a token of overlap between the cue predicted by the system and the gold cue.
 - In "negated(no cue match)", negated is evaluated apart from cue and scope.

 For a full negation to be correct, all elements have to be correct: cue, scope, and negated.

 False negatives are counted either by the system not identifying negation elements present in gold, or by identifying them partially, i.e., not all tokens have been correctly identified or the word forms are incorrect.

 False positives are counted when the system produces a negation element not present in gold.

 True positives are counted when the system produces negation elements exactly as they are in gold.

 Example 1:

 Gold annotation: cue is "un" and  scope is "decided".

 If system identifies "und" as cue and "decided" as scope, it will be counted as false negative for cue and for scope (in scopes cue match), 
 because cue is incorrect.
 If system identifies "un" as cue and "undecided" as scope, cue will count as true positive and scope as false negative.

 Example 2:

 Gold annotation: cue is "un" and  scope is "decided".
 System doesn't have a negation.
 Cue and scope will be false negatives (in scopes cue match).

 Example 3:

 Gold doesn't have a negation.
 System produces a negation, then the negation elements produced by system will count as false positives.

 Example 4:

 Gold annotation: cue is "never", scope is "Holmes entered in the house".
 System output: cue is "never", scope is "entered in the house".
 Cue will be true positive, but scope will be false negative because not all tokens have been produced by system.


 EXAMPLE OF FORMAT EXPECTED

 Without negation: the 8th column is "***" for all tokens.

 Wisteria_ch1	49	0	"	"	PUNC"	(S(NPB*	***
 Wisteria_ch1	49	1	Your	your	PRP$	*	***
 Wisteria_ch1	49	2	telegram	telegram	NN	*NPB)	***
 Wisteria_ch1	49	3	was	be	VBD	(VP*	***
 Wisteria_ch1	49	4	dispatched	dispatch	VBN	(VP*	***
 Wisteria_ch1	49	5	about	about	RB	(QP*	***
 Wisteria_ch1	49	6	one	one	CD	*	***
 Wisteria_ch1	49	7	.	.	PUNC.	*QP)VP)VP)S)	***

 With negation: the columns for negation start at the 8th columns. There will be three columns for each negation (cue, scope, negated). A negation must have at least a negation cue annotated. 

 Wisteria_ch1	217	0	It	it	PRP	(S(S*	_	_	_
 Wisteria_ch1	217	1	is	be	VBZ	(VP*	_	_	_
 Wisteria_ch1	217	2	a	a	DT	(NPB*	_	_	_
 Wisteria_ch1	217	3	lonely	lonely	JJ	*	_	_	_
 Wisteria_ch1	217	4	corner	corner	NN	*	_	_	_
 Wisteria_ch1	217	5	,	,	PUNC,	*NPB)VP)S)	_	_	_
 Wisteria_ch1	217	6	and	and	CC	*	_	_	_
 Wisteria_ch1	217	7	there	there	EX	(S*	_	there	_
 Wisteria_ch1	217	8	is	be	VBZ	(VP*	_	is	_
 Wisteria_ch1	217	9	no	no	DT	(NP(NPB*	no	_	_
 Wisteria_ch1	217	10	house	house	NN	*NPB)	_	house	house
 Wisteria_ch1	217	11	within	within	IN	(PP*	_	within	_
 Wisteria_ch1	217	12	a	a	DT	(NP(NPB*	_	a	_
 Wisteria_ch1	217	13	quarter	quarter	NN	*NPB)	_	quarter	_
 Wisteria_ch1	217	14	of	of	IN	(PP*	_	of	_
 Wisteria_ch1	217	15	a	a	DT	(NP(NPB*	_	a	_
 Wisteria_ch1	217	16	mile	mile	NN	*NPB)	_	mile	_
 Wisteria_ch1	217	17	of	of	IN	(PP*	_	of	_
 Wisteria_ch1	217	18	the	the	DT	(NPB*	_	the	_
 Wisteria_ch1	217	19	spot	spot	NN	*	_	spot	_
 Wisteria_ch1	217	20	.	.	PUNC.	*NPB)PP)NP)PP)NP)PP)NP)VP)S)S)	_	_	_


 


MYEOT
   exit;
 }

if ((defined $opt_h) || ((! defined $opt_g) && (! defined $opt_s)))
{
#  die $usage ;

print<<'EOT';

    *SEM Shared Task 2012 evaluation script for CD-SCO task:

     [perl] eval.cd-sco.pl [OPTIONS] -g <gold standard> -s <system output>

    This script evaluates a system output with respect to a gold standard.

    The two files need to have the same number of lines.

    Optional parameters:
       -h : help:        print this help text and exit
       -r : readme:      print a brief explanation about the evaluation output
EOT
   exit;



}

if (! defined $opt_g)
{
  die "Gold standard file (-g) missing\n" ;
}

if (! defined $opt_s)
{
  die "System output file (-s) missing\n" ;
}




################################################################################
###                              subfunctions                                ###
################################################################################


##################################################
######          SUBROUTINE               #########
###### get info per sentence per line    #########
##################################################

sub get_info_sentence {

### reads a line
### stores columns with negation information (7 to end), 
### in array @neg_cols_g, @neg_cols_p
### g stands for gold file, p for system file

  @tmp_lineg = ();
  @tmp_linep = ();
  
 
  @tmp_lineg = split /\t/, $tmp_lineg;
  @tmp_linep = split /\t/, $tmp_linep;
  
  $max_lineg = $#tmp_lineg;
  $max_linep = $#tmp_linep;
  
  $tmp_neg_g = "";
  $tmp_neg_p = "";
  
  push @POS, $tmp_lineg[5];

  for ($i=7;$i<=$max_lineg;$i++){
    $val = $i / 3;
    if ($i == 7 && $tmp_lineg[$i] eq "***"){
      $tmp_neg_g = "$tmp_lineg[$i]";
    }elsif ($i == 7){
      $tmp_neg_g = "$tmp_lineg[$i] ";
    } elsif ($i == $max_lineg){
      $tmp_neg_g = "$tmp_neg_g$tmp_lineg[$i]";
    } elsif (!( $val =~ /\D+/) ){
      $tmp_neg_g = "$tmp_neg_g$tmp_lineg[$i]\t";
    } else {
      $tmp_neg_g = "$tmp_neg_g$tmp_lineg[$i] ";
    }
  }
  push @neg_cols_g, "$tmp_neg_g";
  
  for ($i=7;$i<=$max_linep;$i++){
    $val = $i / 3;  
    if ($i == 7 && $tmp_linep[$i] eq "***"){
      $tmp_neg_p = "$tmp_linep[$i]";
    }elsif ($i == 7){
      $tmp_neg_p = "$tmp_linep[$i] ";
    } elsif ($i == $max_linep){
      $tmp_neg_p = "$tmp_neg_p$tmp_linep[$i]";
    } elsif (!( $val =~ /\D+/) ){
      $tmp_neg_p = "$tmp_neg_p$tmp_linep[$i]\t";
    } else {
      $tmp_neg_p = "$tmp_neg_p$tmp_linep[$i] ";
    }
  }
  push @neg_cols_p, "$tmp_neg_p";
  
}## end sub

##################################################
######          SUBROUTINE               #########
######        process sentence           #########
##################################################


sub process_sentence {


### processes information in arrays @neg_cols_g, @neg_cols_p
### the arrays contain as main elements as tokens in sentence
### each element of the array contains information per negation separated by tabs, 
### and information of each negation separated by blank space
### _ _ _\t_ _ _

### Information is stored in the arrays below
### They have as many elements as negations there are in the sentence

### arrays of arrays

### @neg_words_g 
### @scope_words_g 
### @scope_words_nopunc_g 
### @negated_words_g 

### @neg_tokens_g 
### @scope_tokens_g 
### @scope_tokens_nopunc_g
### @negated_tokens_g 

### @neg_words_p 
### @scope_words_p 
### @scope_words_nopunc_p
### @negated_words_p 

### @neg_tokens_p 
### @scope_tokens_p 
### @scope_tokens_nopunc_p 
### @negated_tokens_p 

### arrays

### @first_token_negs_g 
### @first_token_negs_p 

### @first_token_negated_g 
### @first_token_negated_p 

### @first_token_scope_g 
### @first_token_scope_p 



my @tmp_neg_cols_g = ();
my @neg_by_neg_cols_g = ();

my @tmp_neg_cols_p = ();
my @neg_by_neg_cols_p = ();

#############################
### 1. Process gold sentence
#############################

$count_sentences++;


### no negations in gold
if ($neg_cols_g[0] eq "***"){
  
  $zero_negs_g = "yes";
  
### negations in gold
} else {

  
  $count_sentences_negation++;
  $zero_negs_g = "no";
  
  #### process gold file
  
  $max_neg_cols_g = $#neg_cols_g;
  #$i count line number in sentence
  for ($i=0;$i<=$max_neg_cols_g;$i++){
    @tmp_neg_cols_g =  split /\t/, $neg_cols_g[$i];
    $max_negs_g = $#tmp_neg_cols_g;
    #$z counts index of negation per line
    for ($z=0;$z<=$max_negs_g;$z++){
      @neg_by_neg_cols_g  = split / /, $tmp_neg_cols_g[$z];

      if ($neg_by_neg_cols_g[0] ne "_"){
	push @{$neg_words_g[$z]}, $neg_by_neg_cols_g[0];
	push @{$neg_tokens_g[$z]}, $i;
      }

      if ($neg_by_neg_cols_g[1] ne "_"){

	if ($POS[$i] =~ /\w/ && $POS[$i] ne "-LRB-" && $POS[$i] ne "-RRB-" ){
	  if ($neg_by_neg_cols_g[1] =~ /^(\w+)\./){
	    $tmp_word = $1;
	  } else {
	    $tmp_word = $neg_by_neg_cols_g[1];
	  }
	  
	  push @{$scope_words_g[$z]}, $tmp_word;
	  #push @{$scope_words_g[$z]}, $neg_by_neg_cols_g[1];
	  push @{$scope_tokens_g[$z]}, $i;
	$total_scope_tokens_g++;
	}

	if ($POS[$i] =~ /\w/ && $POS[$i] ne "-LRB-" && $POS[$i] ne "-RRB-" ){
	  if ($neg_by_neg_cols_g[1] =~ /^(\w+)\./){
	    $tmp_word = $1;
	  } else {
	    $tmp_word = $neg_by_neg_cols_g[1];
	  }
	  push @{$scope_words_nopunc_g[$z]}, $tmp_word;
	  #push @{$scope_words_nopunc_g[$z]}, $neg_by_neg_cols_g[1];
	  push @{$scope_tokens_nopunc_g[$z]}, $i;
	}
      }

      if ($neg_by_neg_cols_g[2] ne "_"){
	push @{$negated_words_g[$z]}, $neg_by_neg_cols_g[2];
	push @{$negated_tokens_g[$z]}, $i;
      }
      
      @neg_by_neg_cols_g = ();


 

    }### $z


    @tmp_neg_cols_g = ();
    
  }
  

   for ($z=0;$z<=$max_negs_g;$z++){
      if ($z >= scalar(@neg_words_g)){
	@{$neg_words_g[$z]} = ();
      }
     if ($z >= scalar(@neg_tokens_g)){
	@{$neg_tokens_g[$z]} = ();
      }
      if ($z >= scalar(@scope_words_g)){
	@{$scope_words_g[$z]} = ();
      }
      if ($z >= scalar(@scope_words_nopunc_g)){
	@{$scope_words_nopunc_g[$z]} = ();
      }
      if ($z >= scalar(@scope_tokens_g)){
	@{$scope_tokens_g[$z]} = ();
      }
     if ($z >= scalar(@scope_tokens_nopunc_g)){
	@{$scope_tokens_nopunc_g[$z]} = ();
      }
      if ($z >= scalar(@negated_words_g)){
	@{$negated_words_g[$z]} = ();
      }
      if ($z >= scalar(@negated_tokens_g)){
	@{$negated_tokens_g[$z]} = ();
      }
      $neg_found_g[$z] = 0;
      $scope_found_g[$z] = 0;
      $negated_found_g[$z] = 0;
      
    }


  for ($z=0;$z<=$max_negs_g;$z++){
    if (@{$neg_tokens_g[$z]}){
      push @first_token_negs_g, $neg_tokens_g[$z][0];
      $cues_g++;
    } else {
      push @first_token_negs_g, "_";
    }
    if (@{$negated_tokens_g[$z]}){
      push @first_token_negated_g, $negated_tokens_g[$z][0];
      $negated_g++;
     } else {
      push @first_token_negated_g, "_";
    }
    if (@{$scope_tokens_g[$z]}){
      push @first_token_scope_g, $scope_tokens_g[$z][0];
      $scopes_g++;
    } else {
      push @first_token_scope_g, "_";
    }
  
  }


}

#############################
### 2. Process system sentence
#############################

### no negation in system
if ($neg_cols_p[0] eq "***"){
  
  $zero_negs_p = "yes";
  
### negations in system
} else {
  
  $zero_negs_p = "no";

  #### process system file
  
  $max_neg_cols_p = $#neg_cols_p;
  #$i count line number in sentence
  for ($i=0;$i<=$max_neg_cols_p;$i++){
    @tmp_neg_cols_p =  split /\t/, $neg_cols_p[$i];
    $max_negs_p = $#tmp_neg_cols_p;
    #$z counts index of negation per line
    for ($z=0;$z<=$max_negs_p;$z++){
      @neg_by_neg_cols_p  = split / /, $tmp_neg_cols_p[$z];
      if ($neg_by_neg_cols_p[0] ne "_"){
	
	push @{$neg_words_p[$z]}, $neg_by_neg_cols_p[0];
	push @{$neg_tokens_p[$z]}, $i;
      }
      if ($neg_by_neg_cols_p[1] ne "_"){

	if ($POS[$i] =~ /\w/ && $POS[$i] ne "-LRB-"&& $POS[$i] ne "-RRB-"){
	  if ($neg_by_neg_cols_p[1] =~ /^(\w+)\./){
	    $tmp_word = $1;
	  } else {
	    $tmp_word = $neg_by_neg_cols_p[1];
	  }
	  
	  push @{$scope_words_p[$z]}, $tmp_word;
	  #push @{$scope_words_p[$z]}, $neg_by_neg_cols_p[1];
	  push @{$scope_tokens_p[$z]}, $i;
	  $total_scope_tokens_p++;
	}
	
	
	if ($POS[$i] =~ /\w/ && $POS[$i] ne "-LRB-"&& $POS[$i] ne "-RRB-"){
	  if ($neg_by_neg_cols_p[1] =~ /^(\w+)\./){
	    $tmp_word = $1;
	  } else {
	    $tmp_word = $neg_by_neg_cols_p[1];
	  }
	  push @{$scope_words_nopunc_p[$z]}, $tmp_word;
	  # push @{$scope_words_nopunc_p[$z]}, $neg_by_neg_cols_p[1];
	  push @{$scope_tokens_nopunc_p[$z]}, $i;
	}
      }
      if ($neg_by_neg_cols_p[2] ne "_"){
	push @{$negated_words_p[$z]}, $neg_by_neg_cols_p[2];
	push @{$negated_tokens_p[$z]}, $i;
      }
      
      @neg_by_neg_cols_p = ();
      
      
    }
    @tmp_neg_cols_p = ();
    
  }
  

  

  for ($z=0;$z<=$max_negs_p;$z++){
      if ($z >= scalar(@neg_words_p)){
	@{$neg_words_p[$z]} = ();
      }
     if ($z >= scalar(@neg_tokens_p)){
	@{$neg_tokens_p[$z]} = ();
      }
      if ($z >= scalar(@scope_words_p)){
	@{$scope_words_p[$z]} = ();
      }
      if ($z >= scalar(@scope_words_nopunc_p)){
	@{$scope_words_nopunc_p[$z]} = ();
      }
      if ($z >= scalar(@scope_tokens_p)){
	@{$scope_tokens_p[$z]} = ();
      }
     if ($z >= scalar(@scope_tokens_nopunc_p)){
	@{$scope_tokens_nopunc_p[$z]} = ();
      }
     if ($z >= scalar(@negated_words_p)){      
	@{$negated_words_p[$z]} = ();
      }
    if ($z >= scalar(@negated_tokens_p)){
	@{$negated_tokens_p[$z]} = ();
      }
      $neg_found_p[$z] = 0;
     $scope_found_p[$z] = 0;
      $negated_found_p[$z] = 0;
      
    }


  ## check that a set of negation columns has at least a negation cue
  for ($z=0;$z<=$max_negs_p;$z++){
    if (!(@{$neg_words_p[$z]})){
      die "Sentence before line $line_number lacks at least a negation cue.\nThe columns for negation where found without cue.\n Fix this before proceedings to evaluate.\n";
     
    }
 
  }



  for ($z=0;$z<=$max_negs_p;$z++){
    if (@{$neg_tokens_p[$z]}){
      push @first_token_negs_p, $neg_tokens_p[$z][0];
      $cues_p++;
    } else {
      push @first_token_negs_p, "_";
    }
    if (@{$negated_tokens_p[$z]}){
      push @first_token_negated_p, $negated_tokens_p[$z][0];
      $negated_p++;
    } else {
      push @first_token_negated_p, "_";
    }
    if (@{$scope_tokens_p[$z]}){
      push @first_token_scope_p, $scope_tokens_p[$z][0];
      $scopes_p++;
    } else {
      push @first_token_scope_p, "_";
    }
  }

 
 

}


} ## end sub

##################################################
######          SUBROUTINE               #########
######     update counts for eval        #########
##################################################

sub update_counts_for_eval {

### Counting number of tp, fp, fn for:
### cue
### scope (tp requires that cue is correct)
### negated (tp requires that cue is correct)
### negated apart (calculated apart from cue and scope, tp does not require correct cue)
### for cue, scope and negated to be correct, both, the tokens and the words or part of words have to be
### correclty identified, else they count as fn
### example 1:
### gold: cue is "un" and  scope is "decided"
### if system identifies "und" as cue and "decided" as scope, it will be counted as false negative for cue, 
### and scope will also be false negative, because cue is incorrect;
### if system identifies "un" as cue and "undecided" as scope, cue will count as true positive
### and scope as false negative;
### example 2:
### gold: cue is "un" and  scope is "decided"
### system doesn't have a negation.
### cue and scope will be false negatives
### example 3:
### gold doesn't have a negation.
### system finds a negation with its scope, then cue and scope will count as false positives
### example 4:
### gold: cue is "never", scope is "Holmes entered in the house"
### system: cue is "never", scope is "entered in the house"
### cue will be true positive, but scope will be false negative because not all tokens have been found by system 


### false negatives are produced either by the system not identifying a negation and its elements present in gold
### or by identifying them incorrectly: not all tokens have been identified or the word forms are incorrect

$st_neg_words_g = "";
$st_scope_words_g = "";
$st_scope_words_nopunc_g = "";
$st_negated_words_g = "";

$st_neg_tokens_g = "";
$st_scope_tokens_g = "";
$st_negated_tokens_g = "";

$st_neg_words_p = "";
$st_scope_words_p = "";
$st_scope_words_nopunc_p = "";
$st_negated_words_p = "";

$st_neg_tokens_p = "";
$st_scope_tokens_p = "";
$st_negated_tokens_p = "";


if (@first_token_negs_g) {
  $max_negs_g = $#first_token_negs_g;
} else {
  $max_negs_g = -1;
}

if (@first_token_negs_p) {
  $max_negs_p = $#first_token_negs_p;
} else {
  $max_negs_p = -1;
}


if (@first_token_negated_g){
  $max_negated_g = $#first_token_negated_g;
} else {
  $max_negated_g =  -1;
}

if (@first_token_negated_p){
  $max_negated_p = $#first_token_negated_p; 
} else {
  $max_negated_p = -1;
}




####################################################
## gold has negations in sentence, system has
## not found negations in sentence
####################################################
if ($zero_negs_g eq "no" && $zero_negs_p eq "yes"){

  $error_found = 1;
  
 
   for ($i=0;$i<=$max_negs_g;$i++){


	if (@{$neg_tokens_g[$i]}){
	 $fn_cue++;
	}

	if (@{$scope_tokens_g[$i]}){
	  $fn_scope++;
	  for ($z=0;$z<=$#{$scope_tokens_g[$i]};$z++){
	    $fn_scope_tokens++;
	  }
	}

	if (@{$negated_tokens_g[$i]}){
	 $fn_negated++;
	}

	$fn_full_negation++;

     
   }


####################################################
## gold and system have negations in sentence
####################################################

} elsif ($zero_negs_g eq "no" && $zero_negs_p eq "no"){


 
  ## $i iterates over negations in gold
  for ($i=0;$i<=$max_negs_g;$i++){

   

    
    #$z iterates over negations in system
    for ($z=0;$z<=$max_negs_p;$z++){
      

        
      
	## udpate variables with the string of negation cue, scope and negated
	## in gold
	
	if (@{$neg_words_g[$i]}){
	  $st_neg_words_g = join(" ",@{$neg_words_g[$i]});
	} else {
	  $st_neg_words_g = "";
	}
	
	if (@{$scope_words_g[$i]}){
	  $st_scope_words_g = join(" ",@{$scope_words_g[$i]});
	} else {
	  $st_scope_words_g = "";
	}

	if (@{$negated_words_g[$i]}){
	  $st_negated_words_g = join(" ",@{$negated_words_g[$i]});
	  
	} else {
	  $st_negated_words_g = "";
	}
	
	if (@{$neg_tokens_g[$i]}){
	  $st_neg_tokens_g = join(" ",@{$neg_tokens_g[$i]});
	} else {
	  $st_neg_tokens_g = "";
	}

	if (@{$scope_tokens_g[$i]}){
	  $st_scope_tokens_g = join(" ",@{$scope_tokens_g[$i]});
	} else {
	  $st_scope_tokens_g = "";
	}
	if (@{$negated_tokens_g[$i]}){
	  $st_negated_tokens_g = join(" ",@{$negated_tokens_g[$i]});
	} else {
	  $st_negated_tokens_g = "";
	}

	## udpate variables with the string of negation cue, scope and negated
	## in system
	
	if (@{$neg_words_p[$z]}){
	  $st_neg_words_p = join(" ",@{$neg_words_p[$z]});
	} else {
	  $st_neg_words_p = "";
	}


	if (@{$scope_words_p[$z]}){
	  $st_scope_words_p = join(" ",@{$scope_words_p[$z]});
	} else {
	  $st_scope_words_p = "";
	}
	if (@{$negated_words_p[$z]}){
	  $st_negated_words_p = join(" ",@{$negated_words_p[$z]});
	} else {
	  $st_negated_words_p = "";
	}
	
	if (@{$neg_tokens_p[$z]}){
	  $st_neg_tokens_p = join(" ",@{$neg_tokens_p[$z]});
	} else {
	  $st_neg_tokens_p = "";
	}
	if (@{$scope_tokens_p[$z]}){
	  $st_scope_tokens_p = join(" ",@{$scope_tokens_p[$z]});
	} else {
	  $st_scope_tokens_p = "";
	}
	if (@{$negated_tokens_p[$z]}){
	  $st_negated_tokens_p = join(" ",@{$negated_tokens_p[$z]});
	} else {
	  $st_negated_tokens_p = "";
	}
	


	$max_neg_tokens_p = $#{$neg_tokens_p[$z]};
	$max_neg_tokens_g = $#{$neg_tokens_g[$i]};
	
	$found = 0;

	for ($y=0;$y<=$max_neg_tokens_g;$y++){
	  for ($x=0;$x<=$max_neg_tokens_p;$x++){
	    	 
	    if ($neg_tokens_g[$i][$y] == $neg_tokens_p[$z][$x] && $neg_found_p[$z] == 0 &&  $neg_found_g[$i] == 0 ){
	      $found = 1;
	      $neg_found_p[$z] = 1;
	      $neg_found_g[$i] = 1;
	      last;
	    }
	  } #for x
	}#for y
	
    
	## if a negation in gold is also found in system
	if ($found ==1){

	  
	  
	  ## update count for scope tokens
	  $max_scope_tokens_p = $#{$scope_tokens_p[$z]};
	  $max_scope_tokens_g = $#{$scope_tokens_g[$i]};
	 

	  ## iterate over gold tokens to find tp and fn
	 
	  for ($y=0;$y<=$max_scope_tokens_g;$y++){
	    $found_scope_token = 0;
	    for ($x=0;$x<=$max_scope_tokens_p;$x++){
	      if ($scope_tokens_g[$i][$y] == $scope_tokens_p[$z][$x]){
		if ($scope_words_g[$i][$y] eq $scope_words_p[$z][$x]){
		  $found_scope_token = 1;
		   
		 
		  
		}
		last;
	      }
	    } #for x
	    
	    if ($found_scope_token == 1){
	      
	      $tp_scope_tokens++;

	    } else {
	      $fn_scope_tokens++;

	    }
	    
	  }#for y
	  

	  ## iterate over system tokens to find fp
	  for ($x=0;$x<=$max_scope_tokens_p;$x++){
	    $found_scope_token = 0;
	    for ($y=0;$y<=$max_scope_tokens_g;$y++){
	      if ($scope_tokens_g[$i][$y] == $scope_tokens_p[$z][$x]){
		if ($scope_words_g[$i][$y] eq $scope_words_p[$z][$x]){
		  $found_scope_token = 1;
		}
		last;
	      }
	    } #for y
	    
	    if ($found_scope_token == 0){
	      $fp_scope_tokens++;
	    

	    }

	    
	  }#for x




	## check whether full negation is correct
	if ($st_neg_tokens_g ne "" && $st_neg_tokens_g eq  $st_neg_tokens_p  && $st_neg_words_g eq  $st_neg_words_p && $st_scope_tokens_g eq $st_scope_tokens_p && $st_scope_words_g eq $st_scope_words_p && $st_negated_tokens_g eq $st_negated_tokens_p && $st_negated_words_g eq $st_negated_words_p){
	  $tp_full_negation++;
	} else {
	  $fn_full_negation++;
	  $error_found = 1;
	  
	}


	## gold cue is correctly identified: both the token number and the word or part of a word
	if ($st_neg_tokens_g ne "" && $st_neg_tokens_g eq  $st_neg_tokens_p  && $st_neg_words_g eq  $st_neg_words_p){
	  $tp_cue++;
	  
	  ########### scope
	  # if no scope was marked for this cue in gold, and system marks it, then it is fp
	  if ($st_scope_tokens_g eq "" && $st_scope_tokens_p ne ""){
	    $fp_scope++;
	    $error_found = 1;
	    
	    ##scope is correctly identified: boh the token numbers and the words or parts of words
	    ## cue needs to have been correctly identified for scope to be counted as correct
	  } elsif ($st_scope_tokens_g ne "" && $st_scope_tokens_g eq $st_scope_tokens_p && $st_scope_words_g eq $st_scope_words_p){
	    $tp_scope++;

	    
	    ## gold marks a scope, in system either the tokens or words are incorrect
	  } elsif ($st_scope_tokens_g ne "" && ($st_scope_tokens_p ne $st_scope_tokens_g || $st_scope_words_p ne $st_scope_words_g)) {
	    $fn_scope++;
	    $error_found = 1;
	    
	  } 
	  
	  ########### negated
	  # if no negated was marked for this cue in gold, and system marks it, then it is fp
	  if ($st_negated_tokens_g eq "" && $st_negated_tokens_p ne ""){
	    $fp_negated++;
	    $error_found = 1;
	    
	    ##negated is correctly identified: boh the token numbers and the words or parts of words
	    ## cue needs to have been correctly identified for negated to be counted as correct
	  } elsif ($st_negated_tokens_g ne "" && $st_negated_tokens_g eq $st_negated_tokens_p && $st_negated_words_g eq $st_negated_words_p){
	    $tp_negated++;
	    
	    ## gold marks a negated, in system either the tokens or words are incorrect
	  } elsif ($st_negated_tokens_g ne "" && ($st_negated_tokens_p ne $st_negated_tokens_g || $st_negated_words_p ne $st_negated_words_g)) {
	    $fn_negated++;
	    $error_found = 1;
	    
	  } 
	  
	  
	  ### well identified token number of negation, but not well identified the word;
	  ### for example, in "unbrushed", "un" is the negation, but not "unbrushed" 
	} elsif ($st_neg_tokens_g ne "" && ($st_neg_tokens_g ne  $st_neg_tokens_p  || $st_neg_words_g ne  $st_neg_words_p)){
	  
	  $fn_cue++;
	  
	  # if ($st_scope_tokens_g ne "" && $st_scope_tokens_p eq ""){
	  if ($st_scope_tokens_g ne ""){
	    $fn_scope++;
	  }
	  
	  #if ($st_negated_tokens_g ne "" && $st_negated_tokens_p eq ""){
	  if ($st_negated_tokens_g ne ""){
	    $fn_negated++;
	  }
	  
	  
	  $error_found = 1;
	  
	  
	}
	

	## gold negation found in system negations; search stops
	last;
	
	## iteration on system negations has finished and gold negation has not been found 
      } elsif ($z==$max_negs_p){


	$error_found = 1;


	if ($st_neg_tokens_g ne ""){
	  $fn_cue++;
	}
	if ($st_scope_tokens_g ne "" ){
	  $fn_scope++;
	  for ($y=0;$y<=$#{$scope_tokens_g[$i]};$y++){
	    $fn_scope_tokens++;
	  }
	}
	if ($st_negated_tokens_g ne ""){
	  $fn_negated++;
	}

	$fn_full_negation++;
      }


    }##for $z
    
  }##for $i


  ### iterate over negations in system; if they are not found in gold, then count false positives
 
 
   
  #$z iterates over negations in system
  for ($z=0;$z<=$max_negs_p;$z++){

    
    ## $i iterates over negations in gold
    for ($i=0;$i<=$max_negs_g;$i++){
      
      

	## udpate variables with the string of the negation cue 
	## in system
	
	if (@{$neg_words_p[$z]}){
	  $st_neg_words_p = join(" ",@{$neg_words_p[$z]});
	} else {
	  $st_neg_words_p = "";
	}

	if (@{$scope_words_p[$z]}){
	  $st_scope_words_p = join(" ",@{$scope_words_p[$z]});
	} else {
	  $st_scope_words_p = "";
	}

	if (@{$negated_words_p[$z]}){
	  $st_negated_words_p = join(" ",@{$negated_words_p[$z]});
	} else {
	  $st_negated_words_p = "";
	}

	
	if (@{$neg_tokens_p[$z]}){
	  $st_neg_tokens_p = join(" ",@{$neg_tokens_p[$z]});
	} else {
	  $st_neg_tokens_p = "";
	}

	if (@{$scope_tokens_p[$z]}){
	  $st_scope_tokens_p = join(" ",@{$scope_tokens_p[$z]});
	} else {
	  $st_scope_tokens_p = "";
	}

	if (@{$negated_tokens_p[$z]}){
	  $st_negated_tokens_p = join(" ",@{$negated_tokens_p[$z]});
	} else {
	  $st_negated_tokens_p = "";
	}

	
	$max_neg_tokens_p = $#{$neg_tokens_p[$z]};
	$max_neg_tokens_g = $#{$neg_tokens_g[$i]};

	$found =0;

  

	for ($x=0;$x<=$max_neg_tokens_p;$x++){
	  for ($y=0;$y<=$max_neg_tokens_g;$y++){  
	    
	    if ($neg_tokens_g[$i][$y] == $neg_tokens_p[$z][$x]  && $neg_found_p[$z] == 1){
	      $found = 1;
	      last;
	    }
	  } #for x
	}#for y
	
    

      ##negation in system is found in gold
      ## this has been treated above 
     
 	if ($found ==1){
	  
	  last;
	 
	  
	  ## negation in system is not found in gold
	} elsif ($i == $max_negs_g ){
	 
	  
	  $error_found = 1;
	  
	  if ($st_neg_tokens_p ne "" ){
	    $fp_cue++;
	  }
	  if ($st_scope_tokens_p ne "" ){
	    $fp_scope++;
	    for ($y=0;$y<=$#{$scope_tokens_p[$z]};$y++){
	      $fp_scope_tokens++;
	     
	    }
	  }
	  if ($st_negated_tokens_p ne ""){
	    $fp_negated++;
	  }
	  
	  
	  $fp_full_negation++;
	  
	  
	}
	
      }##for $i
    
  }##for $z
  
  
####################################################
##  gold doesn't have negations and system has
####################################################

}  elsif ($zero_negs_g eq "yes" && $zero_negs_p eq "no") {

 
  
  	    $error_found = 1;
	   
   for ($z=0;$z<=$max_negs_p;$z++){

	if (@{$neg_tokens_p[$z]}){
	 $fp_cue++;
	}

	if (@{$scope_tokens_p[$z]}){
	  $fp_scope++;
	  for ($y=0;$y<=$#{$scope_tokens_p[$z]};$y++){
	     $fp_scope_tokens++;
	  }
	}
	
	if (@{$negated_tokens_p[$z]}){
	 $fp_negated++;
	 	}

	$fp_full_negation++;

   }
   
 }



update_counts_for_negated_apart();
update_counts_for_scope_apart();


} ## end sub

#######################################################
###### update counts for scope apart from negation cues
#######################################################

sub update_counts_for_scope_apart {


####################################################
## gold has negations in sentence, system has
## not found negations in sentence
####################################################
if ($zero_negs_g eq "no" && $zero_negs_p eq "yes"){

   
   for ($i=0;$i<=$max_negs_g;$i++){

	if (@{$scope_tokens_g[$i]}){
	     $fn_scope_apart++;	
	}

	if (@{$scope_tokens_nopunc_g[$i]}){
	     $fn_scope_nopunc++;	
	}

     
   }


####################################################
## gold and system have negations in sentence
####################################################

} elsif ($zero_negs_g eq "no" && $zero_negs_p eq "no"){

 
  ## $i iterates over negations in gold
  for ($i=0;$i<=$max_negs_g;$i++){

   

    #$z iterates over negations in system
    for ($z=0;$z<=$max_negs_p;$z++){

    
	## udpate variables with the string of negation cue, scope and negated
	## in gold
	
	if (@{$neg_words_g[$i]}){
	  $st_neg_words_g = join(" ",@{$neg_words_g[$i]});
	} else {
	  $st_neg_words_g = "";
	}
	
	if (@{$scope_words_g[$i]}){
	  $st_scope_words_g = join(" ",@{$scope_words_g[$i]});
	} else {
	  $st_scope_words_g = "";
	}

	if (@{$scope_words_nopunc_g[$i]}){
	  $st_scope_words_nopunc_g = join(" ",@{$scope_words_nopunc_g[$i]});
	} else {
	  $st_scope_words_nopunc_g = "";
	}
	
	if (@{$neg_tokens_g[$i]}){
	  $st_neg_tokens_g = join(" ",@{$neg_tokens_g[$i]});
	} else {
	  $st_neg_tokens_g = "";
	}

	if (@{$scope_tokens_g[$i]}){
	  $st_scope_tokens_g = join(" ",@{$scope_tokens_g[$i]});
	} else {
	  $st_scope_tokens_g = "";
	}


	if (@{$scope_tokens_nopunc_g[$i]}){
	  $st_scope_tokens_nopunc_g = join(" ",@{$scope_tokens_nopunc_g[$i]});
	} else {
	  $st_scope_tokens_nopunc_g = "";
	}


	## udpate variables with the string of negation cue, scope and negated
	## in system
	
	if (@{$neg_words_p[$z]}){
	  $st_neg_words_p = join(" ",@{$neg_words_p[$z]});
	} else {
	  $st_neg_words_p = "";
	}

	if (@{$scope_words_p[$z]}){
	  $st_scope_words_p = join(" ",@{$scope_words_p[$z]});
	} else {
	  $st_scope_words_p = "";
	}

	if (@{$scope_words_nopunc_p[$z]}){
	  $st_scope_words_nopunc_p = join(" ",@{$scope_words_nopunc_p[$z]});
	} else {
	  $st_scope_words_nopunc_p = "";
	}

	if (@{$neg_tokens_p[$z]}){
	  $st_neg_tokens_p = join(" ",@{$neg_tokens_p[$z]});
	} else {
	  $st_neg_tokens_p = "";
	}
	if (@{$scope_tokens_p[$z]}){
	  $st_scope_tokens_p = join(" ",@{$scope_tokens_p[$z]});
	} else {
	  $st_scope_tokens_p = "";
	}

	if (@{$scope_tokens_nopunc_p[$z]}){
	  $st_scope_tokens_nopunc_p = join(" ",@{$scope_tokens_nopunc_p[$z]});
	} else {
	  $st_scope_tokens_nopunc_p = "";
	}


	$max_neg_tokens_p = $#{$neg_tokens_p[$z]};
	$max_neg_tokens_g = $#{$neg_tokens_g[$i]};

	
	$found = 0;


	for ($y=0;$y<=$max_neg_tokens_g;$y++){
	  for ($x=0;$x<=$max_neg_tokens_p;$x++){

	    if ($neg_tokens_g[$i][$y] == $neg_tokens_p[$z][$x]  && $scope_found_p[$z] == 0 &&  $scope_found_g[$i] == 0){
	      $found = 1;
	      $scope_found_p[$z] = 1;
	      $scope_found_g[$i] = 1;
	      last;
	    }
	  } #for x
	}#for y
	
    
	## if a negation in gold is also found in system
	if ($found ==1){

  
	  ########### scope apart #####################
	  # if no scope was marked for this cue in gold, and system marks it, then it is fp
	  if ($st_scope_tokens_g eq "" && $st_scope_tokens_p ne ""){
	    $fp_scope_apart++;
	    
	    
	    ##scope is correctly identified: boh the token numbers and the words or parts of words
	    ## cue needs to have been correctly identified for scope to be counted as correct
	  } elsif ($st_scope_tokens_g ne "" && $st_scope_tokens_g eq $st_scope_tokens_p && $st_scope_words_g eq $st_scope_words_p){
	    $tp_scope_apart++;

	    
	    ## gold marks a scope, in system either the tokens or words are incorrect
	  } elsif ($st_scope_tokens_g ne "" && ($st_scope_tokens_p ne $st_scope_tokens_g || $st_scope_words_p ne $st_scope_words_g)) {
	    $fn_scope_apart++;
	  
	    
	  } 
	  

	  ########### scope no punctuation #####################
	  # if no scope was marked for this cue in gold, and system marks it, then it is fp
	  if ($st_scope_tokens_nopunc_g eq "" && $st_scope_tokens_nopunc_p ne ""){
	    $fp_scope_nopunc++;
	    
	    
	    ##scope is correctly identified: boh the token numbers and the words or parts of words
	    ## cue needs to have been correctly identified for scope to be counted as correct
	  } elsif ($st_scope_tokens_nopunc_g ne "" && $st_scope_tokens_nopunc_g eq $st_scope_tokens_nopunc_p && $st_scope_words_nopunc_g eq $st_scope_words_nopunc_p){
	    $tp_scope_nopunc++;

	    
	    ## gold marks a scope, in system either the tokens or words are incorrect
	  } elsif ($st_scope_tokens_nopunc_g ne "" && ($st_scope_tokens_nopunc_p ne $st_scope_tokens_nopunc_g || $st_scope_words_nopunc_p ne $st_scope_words_nopunc_g)) {
	    $fn_scope_nopunc++;
	    
	  } 

	  
       	## gold negation found in system negations; search in system negations stops
	last;

	
	## iteration on system negations has finished and gold negation has not been found 
      } elsif ($z==$max_negs_p ){


 	if ($st_scope_tokens_g ne "" ){
 	  $fn_scope_apart++;
 	}

	if ($st_scope_tokens_nopunc_g ne "" ){
 	  $fn_scope_nopunc++;
	}

		
      }



    }##for $z
    
  }##for $i


  ### iterate over negations in system; if they are not found in gold, then count false positives
 
 
  #$z iterates over negations in system
  for ($z=0;$z<=$max_negs_p;$z++){
    

    ## $i iterates over negations in gold
    for ($i=0;$i<=$max_negs_g;$i++){
      
	## udpate variables with the string of the negation cue 
	## in system
	
	if (@{$neg_words_p[$z]}){
	  $st_neg_words_p = join(" ",@{$neg_words_p[$z]});
	} else {
	  $st_neg_words_p = "";
	}

	if (@{$scope_words_p[$z]}){
	  $st_scope_words_p = join(" ",@{$scope_words_p[$z]});
	} else {
	  $st_scope_words_p = "";
	}

	if (@{$scope_words_nopunc_p[$z]}){
	  $st_scope_words_nopunc_p = join(" ",@{$scope_words_nopunc_p[$z]});
	} else {
	  $st_scope_words_nopunc_p = "";
	}


	if (@{$negated_words_p[$z]}){
	  $st_negated_words_p = join(" ",@{$negated_words_p[$z]});
	} else {
	  $st_negated_words_p = "";
	}
	
	if (@{$neg_tokens_p[$z]}){
	  $st_neg_tokens_p = join(" ",@{$neg_tokens_p[$z]});
	} else {
	  $st_neg_tokens_p = "";
	}

	if (@{$scope_tokens_p[$z]}){
	  $st_scope_tokens_p = join(" ",@{$scope_tokens_p[$z]});
	} else {
	  $st_scope_tokens_p = "";
	}

	if (@{$scope_tokens_nopunc_p[$z]}){
	  $st_scope_tokens_nopunc_p = join(" ",@{$scope_tokens_nopunc_p[$z]});
	} else {
	  $st_scope_tokens_nopunc_p = "";
	}

	if (@{$negated_tokens_p[$z]}){
	  $st_negated_tokens_p = join(" ",@{$negated_tokens_p[$z]});
	} else {
	  $st_negated_tokens_p = "";
	}
	
	$max_neg_tokens_p = $#{$neg_tokens_p[$z]};
	$max_neg_tokens_g = $#{$neg_tokens_g[$i]};

	$found =0;

	for ($x=0;$x<=$max_neg_tokens_p;$x++){
	  for ($y=0;$y<=$max_neg_tokens_g;$y++){    
	    if ($neg_tokens_g[$i][$y] == $neg_tokens_p[$z][$x]  && $scope_found_p[$z] == 1){
	      $found = 1;
	      last;
	    }
	  } #for x
	}#for y
	
    

      ##negation in system is found in gold
      ## this has been treated above 
 	if ($found ==1){
	last;
  

      ## negation in system is not found in gold
      } elsif ($i == $max_negs_g ){

	
	if ($st_scope_tokens_p ne "" ){
	  $fp_scope_apart++;
	}

	if ($st_scope_tokens_nopunc_p ne "" ){
	  $fp_scope_nopunc++;
	 
	}


	
       }
    }##for $i
    
  }##for $z

  
  

####################################################
##  gold doesn't have negations and system has
####################################################

}  elsif ($zero_negs_g eq "yes" && $zero_negs_p eq "no") {

  	     
  for ($z=0;$z<=$max_negs_p;$z++){
    
    
    if (@{$scope_tokens_p[$z]}){
      $fp_scope_apart++;
      
    }
    
    if (@{$scope_tokens_nopunc_p[$z]}){
      $fp_scope_nopunc++;  
    }
    
  }
  
}
  

}

#######################################################
###### update counts for negated apart from negation cues
#######################################################

sub update_counts_for_negated_apart {

 
    ####################################################
    ## gold has negations in sentence, system not
    ####################################################
  if ($zero_negs_g eq "no" && $zero_negs_p eq "yes"){
    
    for ($i=0;$i<=$max_negated_g;$i++){
      
      
      if (@{$negated_tokens_g[$i]}){
	$fn_negated_apart++;
      }
      
      
    }
    
    ####################################################
    ## gold and system have negations in sentence
    ####################################################
    
  } elsif ($zero_negs_g eq "no" && $zero_negs_p eq "no"){
    
    
 
     ### iterate over negations in gold 
     ## $i iterates over negations in gold
     for ($i=0;$i<=$max_negated_g;$i++){
     
  
       #$z iterates over negations in system
       for ($z=0;$z<=$max_negated_p;$z++){

 
	 ## update string of negated in gold
 	  if (@{$negated_words_g[$i]}){
 	    $st_negated_words_g = join(" ",@{$negated_words_g[$i]});
 	  } else {
 	    $st_negated_words_g = "";
 	  }
	  
	  
 	  if (@{$negated_tokens_g[$i]}){
 	    $st_negated_tokens_g = join(" ",@{$negated_tokens_g[$i]});
 	  } else {
 	    $st_negated_tokens_g = "";
 	  }
	  
 	   ## update string of negated in system
 	  if (@{$negated_words_p[$z]}){
 	    $st_negated_words_p = join(" ",@{$negated_words_p[$z]});
 	  } else {
 	    $st_negated_words_p = "";
 	  }
	  
	  
 	  if (@{$negated_tokens_p[$z]}){
 	    $st_negated_tokens_p = join(" ",@{$negated_tokens_p[$z]});
 	  } else {
 	    $st_negated_tokens_p = "";
 	  }


	 
	  $max_negated_tokens_p = $#{$negated_tokens_p[$z]};
	  $max_negated_tokens_g = $#{$negated_tokens_g[$i]};


	$found =0;


	 
	  for ($y=0;$y<=$max_negated_tokens_g;$y++){
	    for ($x=0;$x<=$max_negated_tokens_p;$x++){
	     
	      if ($negated_tokens_g[$i][$y] == $negated_tokens_p[$z][$x]  && $negated_found_p[$z] == 0 &&  $negated_found_g[$i] == 0 ){
		$found = 1;
		$negated_found_p[$z] = 1;
		$negated_found_g[$i] = 1;
		last;
	      
	    }
	      
	    } #for x
	  }#for y

	
    
	## if a negation in gold is also found in system
	if ($found ==1){	  

	  
 	  ## negated is correctly identified
 	  ## both token number and word or part of word have to be correctly identified
	 
 	  if ($st_negated_tokens_g ne "" && $st_negated_tokens_g eq $st_negated_tokens_p && $st_negated_words_g eq $st_negated_words_p){
 	    $tp_negated_apart++;
	    last;	    
	 	    
 	    ## if negated is not correctly identified
 	  } elsif ($st_negated_tokens_g ne "" && ($st_negated_tokens_p ne $st_negated_tokens_g || $st_negated_words_p ne $st_negated_words_g)){

	    ## we need to check whether we are comparing negated elements of the same negation
	    $found = 0;
	    for ($y=0;$y<=$max_negated_words_g;$y++){
	      for ($x=0;$x<=$max_negated_words_p;$x++){
		
		if ($negated_words_g[$i][$y] == $negated_words_p[$z][$x]){
		  $found = 1;
		  last;
		  
		}
		
	      } #for x
	    }#for y
	    
	    if ($found ==1){   
	      $fn_negated_apart++;
	    }

 	  }
	  
	  ## iteration over negations in system finishes without finding gold negation
	} elsif ($z==$max_negated_p){
	  
	  if ($st_negated_tokens_g ne ""){

	    $fn_negated_apart++;
	   	    
	  }
	  
	}
	
     }##for $z
    
   }##for $i
    
    
    ### iterate over negations in system; 
    ### if they are not found in gold, then count false positives
    
 
    #$z iterates over negations in system
    for ($z=0;$z<=$max_negated_p;$z++){
      
  
      ## $i iterates over negations in gold
      for ($i=0;$i<=$max_negated_g;$i++){
	

	   ## update string of negated in system
 	  if (@{$negated_words_p[$z]}){
 	    $st_negated_words_p = join(" ",@{$negated_words_p[$z]});
 	  } else {
 	    $st_negated_words_p = "";
 	  }
	  
	  
 	  if (@{$negated_tokens_p[$z]}){
 	    $st_negated_tokens_p = join(" ",@{$negated_tokens_p[$z]});
 	  } else {
 	    $st_negated_tokens_p = "";
 	  }


	$max_negated_tokens_p = $#{$negated_tokens_p[$z]};
	$max_negated_tokens_g = $#{$negated_tokens_g[$i]};
	
	$found =0;

	  
	  for ($x=0;$x<=$max_negated_tokens_p;$x++){
	    for ($y=0;$y<=$max_negated_tokens_g;$y++){
	      
	      if ($negated_tokens_g[$i][$y] == $negated_tokens_p[$z][$x]  && $negated_found_p[$z] == 1){
		$found = 1;
		last;
	      }
	    } #for x
	  }#for y
	  
	  
	  ## if a negation in gold is also found in system
	  if ($found ==1){
	
	  last;
	
	
	## negation in system is not found in gold
	}elsif ($i == $max_negated_g){

	  if ($st_negated_tokens_p ne ""){
	    $fp_negated_apart++;

	  }
	  
	  
	}
      }##for $i
      
    }##for $z
    
    
    ####################################################
    ##  gold doesn't have negations and system has
    ####################################################
    
  }  elsif ($zero_negs_g eq "yes" && $zero_negs_p eq "no") {
    
    
    for ($z=0;$z<=$max_negated_p;$z++){
      
      
      if (@{$negated_tokens_p[$z]}){
	$fp_negated_apart++;

      }
      
     
    }
    
  }
  
  

} ## end sub



################################################################################
###                                  main                                    ###
################################################################################

#######################################################################
##### 1. check that GOLD file and SYSTEM file have the same sentences
##### evaluation does not proceed if sentences are different
#######################################################################

open (GOLD, "<$opt_g") || die "Could not open gold standard file $opt_g\n" ;
open (SYSTEM,  "<$opt_s") || die "Could not open system output file $opt_s\n" ;

while(<GOLD>) {
  chomp;
  $tmp_lineg = $_;
  $tmp_linep = <SYSTEM>;
  chomp $tmp_linep;
  $line_number++;

  @tmp_lineg = ();
  @tmp_linep = ();

  if ($tmp_lineg eq ""){
    if ($tmp_linep ne ""){
      die "Blank line in GOLD file is not blank line in SYSTEM file";
    }

  } else {

    @tmp_lineg = split /\t/, $tmp_lineg;
    @tmp_linep = split /\t/, $tmp_linep;
    
    if ($tmp_linep[0] ne $tmp_lineg[0] || $tmp_linep[1] ne $tmp_lineg[1] || $tmp_linep[2] ne $tmp_lineg[2] || $tmp_linep[3] ne $tmp_lineg[3]){
      print STDERR "ATTENTION: mismatch between lines of GOLD file and SYSTEM file\n";
      print STDERR "In file: $tmp_lineg[0], sentence: $tmp_lineg[1], word: $tmp_lineg[2]\n";
      print STDERR "This needs to be fixed before evaluating\n";
      
      die "Process ended\n"
    
    }
  }
  
}

$line_number = 0;
@tmp_lineg = ();
@tmp_linep = ();
$tmp_lineg = "";
$tmp_linep = "";

close(GOLD);
close(SYSTEM);


#################################################################################
##### 2.1 Check that SYSTEM file annotates sentences without negation consistently 
##### All tokens in column 7 need to have "***"
##### 2.2 Check that all tokens of the same sentence have the same
##### number of columns and that the number of columns is 
##### either 7 (starting by 0) or, if larger, divisible by 3
#######################################################################


open (SYSTEM,  "<$opt_s") || die "Could not open system output file $opt_s\n" ;


my @col7_p = ();
$line_number = 0;
my $max_tmp_linep = -1;
my $max_col7_p = -1;

while(<SYSTEM>) {
  $line_number++;
  $tmp_linep = $_;
  chomp $tmp_linep;
  if ($tmp_linep eq ""){
    $max_col7_p = $#col7_p;
    for ($i = 0;$i<$max_col7_p;$i++){
      if ($col7_p[$i] eq "***" &&  $i>0){
	if ($col7_p[$i-1] ne "***"){
	  die "Inconsistency detected in column 7 of SYSTEM's file\nAll tokens should have value *** for this column\nError in sentence that finishes before line number $line_number, token $i-1\nFix this before proceeding with evaluation";
	}
	if ($col7_p[$i+1] ne "***"){
	  die "Inconsistency detected in column 7 of SYSTEM's file\nAll tokens should have value *** for this column\nError in sentence that finishes before line number $line_number, token $i+1\nFix this before proceeding with evaluation";
	}
      } 
      
    }
    
    @col7_p = ();
    $max_tmp_linep = -1;
  } else{
    @tmp_linep = split /\t/, $tmp_linep;
    push @col7_p, $tmp_linep[7];
    if ($max_tmp_linep == -1){
      $max_tmp_linep = $#tmp_linep;
    }
    if ($max_tmp_linep ne $#tmp_linep){
      die "Inconsistency detected in the number of columns at line number $line_number\nAll tokens in a sentence should have the same number of columns";
    }
    if ($max_tmp_linep != 7){
      $val = $max_tmp_linep  / 3;
      if ( $val =~ /\D+/ ){
	die "Incorrect number of columns in line number $line_number\nThere should be 3 columns per negation cue\nFix this before proceeding to evaluation";
      }
    }
  }
}

$line_number = 0;
$tmp_linep = "";
$max_tmp_linep = -1;

close(SYSTEM);



#######################################################################
###### 3. evaluate
#######################################################################

open (GOLD, "<$opt_g") || die "Could not open gold standard file $opt_g\n" ;
open (SYSTEM,  "<$opt_s") || die "Could not open system output file $opt_s\n" ;


while(<GOLD>) {


  chomp;
  $tmp_lineg = $_;
  $tmp_linep = <SYSTEM>;
  chomp $tmp_linep;
  $line_number++;

  
  ### if line is blank or if end of file is reached
  ### process sentence 
  if ($tmp_lineg eq "" || eof(SYSTEM) || eof(GOLD)){
    
    process_sentence();
    update_counts_for_eval();
    
    if ($error_found == 1){
        #print "$line_number $tmp_lineg\n $tmp_linep\n";
      $count_error_sentences++;
    }

    if ($error_found == 1 && $zero_negs_g eq "no" ){
      $count_error_sentences_negation++;
    }

    ###################################
    ## initialize variables 
    ## before processing next sentence
    @POS = ();

    @neg_cols_g = ();
    @neg_cols_p = ();

    @first_token_negs_g = ();
    @first_token_negated_g = ();
    @first_token_scope_g = ();
    
    @neg_found_g = ();

    @neg_words_g = ();
    @scope_words_g = ();
    @scope_words_nopunc_g = ();
    @negated_words_g = ();
    
    @neg_tokens_g = ();
    @scope_tokens_g = ();
    @scope_tokens_nopunc_g = ();
    @negated_tokens_g = ();
    

    @first_token_negs_p = ();
    @first_token_negated_p = ();
    @first_token_scope_p = ();
    
    @neg_found_p = ();

    @neg_words_p = ();
    @scope_words_p = ();
    @scope_words_nopunc_p = ();
    @negated_words_p = ();
    
    @neg_tokens_p = ();
    @scope_tokens_p = ();
    @scope_tokens_nopunc_p = ();
    @negated_tokens_p = ();

    @negated_found_g = ();
    @negated_found_p = ();
    
    @scope_found_g = ();
    @scope_found_p = ();

    $zero_negs_p = "";
    $zero_negs_g = "";

    $error_found = 0;
    ###################################
 
    ## get information about sentence
  } else {
    get_info_sentence();
  }

}


######### calculate F measures

my $precision_cue;
my $precision_cue_b;
my $recall_cue;
my $f1_cue;
my $f1_cue_b;

my $precision_scope;
my $precision_scope_b;
my $recall_scope;
my $f1_scope;
my $f1_scope_b;

my $precision_scope_tokens;
my $recall_scope_tokens;
my $f1_scope_tokens;


my $precision_negated_apart;
my $precision_negated_apart_b;
my $recall_negated_apart;
my $f1_negated_apart;
my $f1_negated_apart_b;

my $precision_scope_apart;
my $precision_scope_apart_b;
my $recall_scope_apart;
my $f1_scope_apart;
my $f1_scope_apart_b;


my $precision_scope_nopunc;
my $precision_scope_nopunc_b;
my $recall_scope_nopunc;
my $f1_scope_nopunc;
my $f1_scope_nopunc_b;

my $precision_full_negation;
my $precision_full_negation_b;
my $recall_full_negation;
my $f1_full_negation;
my $f1_full_negation_b;

my $perc_error_sentences;
my $perc_error_negation_sentences;
my $perc_correct_sentences;
my $perc_correct_negation_sentences;



####### cues
if ($tp_cue + $fp_cue){
   $precision_cue = sprintf("%.2f",($tp_cue / ($tp_cue + $fp_cue)) * 100);
 } else {
   $precision_cue = sprintf("%.2f",0.00);
 }

if ($cues_p){
   $precision_cue_b = sprintf("%.2f",($tp_cue / ($cues_p)) * 100);
 } else {
   $precision_cue_b = sprintf("%.2f",0.00);
 }

if ($tp_cue + $fn_cue){
  $recall_cue =  sprintf("%.2f",($tp_cue / ($tp_cue + $fn_cue)) * 100);
} else {
  $recall_cue = sprintf("%.2f",0.00);
}

if ($precision_cue + $recall_cue){
  $f1_cue =   sprintf("%.2f",(2 * $precision_cue * $recall_cue) / ($precision_cue + $recall_cue));
} else {
  $f1_cue = sprintf("%.2f",0.00);
}

if ($precision_cue_b + $recall_cue){
  $f1_cue_b =   sprintf("%.2f",(2 * $precision_cue_b * $recall_cue) / ($precision_cue_b + $recall_cue));
} else {
  $f1_cue_b = sprintf("%.2f",0.00);
}

###### scopes

if ($tp_scope + $fp_scope){
  $precision_scope = sprintf("%.2f",($tp_scope / ($tp_scope + $fp_scope)) * 100 );
} else {
  $precision_scope = sprintf("%.2f",0.00);
}

if ($scopes_p){
  $precision_scope_b = sprintf("%.2f",($tp_scope / ($scopes_p)) * 100 );
} else {
  $precision_scope_b = sprintf("%.2f",0.00);
}


if ($tp_scope + $fn_scope){
  $recall_scope = sprintf("%.2f",($tp_scope / ($tp_scope + $fn_scope)) * 100);
} else {
  $recall_scope =sprintf("%.2f",0.00);
}

if ($precision_scope + $recall_scope){
  $f1_scope =  sprintf("%.2f",(2 * $precision_scope * $recall_scope) / ($precision_scope + $recall_scope));
} else {
  $f1_scope =  sprintf("%.2f",0.00);
}


if ($precision_scope_b + $recall_scope){
  $f1_scope_b =  sprintf("%.2f",(2 * $precision_scope_b * $recall_scope) / ($precision_scope_b + $recall_scope));
} else {
  $f1_scope_b =  sprintf("%.2f",0.00);
}


###### scopes apart

if ($tp_scope_apart + $fp_scope_apart){
  $precision_scope_apart = sprintf("%.2f",($tp_scope_apart / ($tp_scope_apart + $fp_scope_apart)) * 100 );
} else {
  $precision_scope_apart = sprintf("%.2f",0.00);
}

if ($scopes_p){
  $precision_scope_apart_b = sprintf("%.2f",($tp_scope_apart / $scopes_p) * 100 );
} else {
  $precision_scope_apart_b = sprintf("%.2f",0.00);
}

if ($tp_scope_apart + $fn_scope_apart){
  $recall_scope_apart = sprintf("%.2f",($tp_scope_apart / ($tp_scope_apart + $fn_scope_apart)) * 100);
} else {
  $recall_scope_apart =sprintf("%.2f",0.00);
}

if ($precision_scope_apart + $recall_scope_apart){
  $f1_scope_apart =  sprintf("%.2f",(2 * $precision_scope_apart * $recall_scope_apart) / ($precision_scope_apart + $recall_scope_apart));
} else {
  $f1_scope_apart =  sprintf("%.2f",0.00);
}

if ($precision_scope_apart_b + $recall_scope_apart){
  $f1_scope_apart_b =  sprintf("%.2f",(2 * $precision_scope_apart_b * $recall_scope_apart) / ($precision_scope_apart_b + $recall_scope_apart));
} else {
  $f1_scope_apart_b =  sprintf("%.2f",0.00);
}

###### scopes nopunc

if ($tp_scope_nopunc + $fp_scope_nopunc){
  $precision_scope_nopunc = sprintf("%.2f",($tp_scope_nopunc / ($tp_scope_nopunc + $fp_scope_nopunc)) * 100 );
} else {
  $precision_scope_nopunc = sprintf("%.2f",0.00);
}

if ($scopes_p){
  $precision_scope_nopunc_b = sprintf("%.2f",($tp_scope_nopunc / $scopes_p) * 100 );
} else {
  $precision_scope_nopunc_b = sprintf("%.2f",0.00);
}

if ($tp_scope_nopunc + $fn_scope_nopunc){
  $recall_scope_nopunc = sprintf("%.2f",($tp_scope_nopunc / ($tp_scope_nopunc + $fn_scope_nopunc)) * 100);
} else {
  $recall_scope_nopunc =sprintf("%.2f",0.00);
}

if ($precision_scope_nopunc + $recall_scope_nopunc){
  $f1_scope_nopunc =  sprintf("%.2f",(2 * $precision_scope_nopunc * $recall_scope_nopunc) / ($precision_scope_nopunc + $recall_scope_nopunc));
} else {
  $f1_scope_nopunc =  sprintf("%.2f",0.00);
}

if ($precision_scope_nopunc_b + $recall_scope_nopunc){
  $f1_scope_nopunc_b =  sprintf("%.2f",(2 * $precision_scope_nopunc_b * $recall_scope_nopunc) / ($precision_scope_nopunc_b + $recall_scope_nopunc));
} else {
  $f1_scope_nopunc_b =  sprintf("%.2f",0.00);
}


###### scope tokens
if ($tp_scope_tokens + $fp_scope_tokens) {
  $precision_scope_tokens = sprintf("%.2f",($tp_scope_tokens / ($tp_scope_tokens + $fp_scope_tokens)) * 100 );
} else {
  $precision_scope_tokens = sprintf("%.2f",0.00);
}

if ($tp_scope_tokens + $fn_scope_tokens){
  $recall_scope_tokens = sprintf("%.2f",($tp_scope_tokens / ($tp_scope_tokens + $fn_scope_tokens)) * 100);
} else {
  $recall_scope_tokens = sprintf("%.2f",0.00);
}

if ($precision_scope_tokens + $recall_scope_tokens){
$f1_scope_tokens =  sprintf("%.2f",(2 * $precision_scope_tokens * $recall_scope_tokens) / ($precision_scope_tokens + $recall_scope_tokens));
} else {
$f1_scope_tokens = sprintf("%.2f",0.00);
}


###### negated apart
if ($tp_negated_apart + $fp_negated_apart){
$precision_negated_apart = sprintf("%.2f",($tp_negated_apart / ($tp_negated_apart + $fp_negated_apart)) * 100);
} else {
$precision_negated_apart = sprintf("%.2f",0.00);
}

if ($negated_p){
$precision_negated_apart_b = sprintf("%.2f",($tp_negated_apart / $negated_p) * 100);
} else {
$precision_negated_apart_b = sprintf("%.2f",0.00);
}

if ($tp_negated_apart + $fn_negated_apart){
$recall_negated_apart = sprintf("%.2f",($tp_negated_apart / ($tp_negated_apart + $fn_negated_apart)) * 100);
} else {
$recall_negated_apart = sprintf("%.2f",0.00);
}

if ($precision_negated_apart + $recall_negated_apart){
  $f1_negated_apart = sprintf("%.2f", (2 * $precision_negated_apart * $recall_negated_apart) / ($precision_negated_apart + $recall_negated_apart));
} else {
  $f1_negated_apart = sprintf("%.2f",0.00);
}

if ($precision_negated_apart_b + $recall_negated_apart){
  $f1_negated_apart_b = sprintf("%.2f", (2 * $precision_negated_apart_b * $recall_negated_apart) / ($precision_negated_apart_b + $recall_negated_apart));
} else {
  $f1_negated_apart_b = sprintf("%.2f",0.00);
}

##### full negation
if ($tp_full_negation + $fp_full_negation){
$precision_full_negation = sprintf("%.2f",($tp_full_negation / ($tp_full_negation + $fp_full_negation)) * 100);
} else {
$precision_full_negation = sprintf("%.2f",0.00);
}


if ($cues_p){
$precision_full_negation_b = sprintf("%.2f",($tp_full_negation / $cues_p) * 100);
} else {
$precision_full_negation_b = sprintf("%.2f",0.00);
}

if ($tp_full_negation + $fn_full_negation){
$recall_full_negation = sprintf("%.2f",($tp_full_negation / ($tp_full_negation + $fn_full_negation)) * 100);
} else {
$recall_full_negation = sprintf("%.2f",0.00);
}


if ($precision_full_negation + $recall_full_negation){
$f1_full_negation = sprintf("%.2f", (2 * $precision_full_negation * $recall_full_negation) / ($precision_full_negation + $recall_full_negation));
} else {
$f1_full_negation =sprintf("%.2f",0.00);
}


if ($precision_full_negation_b + $recall_full_negation){
$f1_full_negation_b = sprintf("%.2f", (2 * $precision_full_negation_b * $recall_full_negation) / ($precision_full_negation_b + $recall_full_negation));
} else {
$f1_full_negation_b =sprintf("%.2f",0.00);
}

##### percentage sentences

$perc_error_sentences =  sprintf("%.2f",($count_error_sentences * 100) /  $count_sentences);
if ($count_sentences_negation){
    $perc_error_negation_sentences =  sprintf("%.2f",($count_error_sentences_negation * 100) /  $count_sentences_negation);
} else {
    $perc_error_negation_sentences =  sprintf("%.2f",0.00);
}
$perc_correct_sentences = sprintf("%.2f",100 - $perc_error_sentences);
$perc_correct_negation_sentences = sprintf("%.2f",100 - $perc_error_negation_sentences);


######### print results

print "---------------------------+------+--------+------+------+------+---------------+------------+---------\n";
print "                            | gold | system | tp   | fp   | fn   | precision (%) | recall (%) | F1  (%) \n";
print "----------------------------+------+--------+------+------+------+---------------+------------+---------\n";


printf ("Cues: %28d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $cues_g, $cues_p, $tp_cue, $fp_cue, $fn_cue,  $precision_cue, $recall_cue, $f1_cue);
printf ("Scopes(cue match): %15d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $scopes_g, $scopes_p, $tp_scope, $fp_scope, $fn_scope,  $precision_scope, $recall_scope, $f1_scope);
printf ("Scopes(no cue match): %12d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $scopes_g, $scopes_p, $tp_scope, $fp_scope_apart, $fn_scope_apart,  $precision_scope_apart, $recall_scope_apart, $f1_scope_apart);
#printf ("Scopes(no cue match, no punc): %3d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $scopes_g, $scopes_p, $tp_scope, $fp_scope_nopunc, $fn_scope_nopunc,  $precision_scope_nopunc, $recall_scope_nopunc, $f1_scope_nopunc);
printf ("Scope tokens(no cue match): %6d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $total_scope_tokens_g, $total_scope_tokens_p, $tp_scope_tokens, $fp_scope_tokens, $fn_scope_tokens, $precision_scope_tokens, $recall_scope_tokens, $f1_scope_tokens);
printf ("Negated(no cue match): %11d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $negated_g, $negated_p, $tp_negated_apart, $fp_negated_apart, $fn_negated_apart,  $precision_negated_apart, $recall_negated_apart, $f1_negated_apart);
printf ("Full negation: %19d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $cues_g, $cues_p, $tp_full_negation, $fp_full_negation, $fn_full_negation,  $precision_full_negation, $recall_full_negation, $f1_full_negation);
print "---------------------------+------+--------+------+------+------+---------------+------------+---------\n";
printf ("Cues B: %26d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $cues_g, $cues_p, $tp_cue, $fp_cue, $fn_cue,  $precision_cue_b, $recall_cue, $f1_cue_b);
printf ("Scopes B (cue match): %12d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $scopes_g, $scopes_p, $tp_scope, $fp_scope, $fn_scope,  $precision_scope_b, $recall_scope, $f1_scope_b);
printf ("Scopes B (no cue match): %9d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $scopes_g, $scopes_p, $tp_scope, $fp_scope_apart, $fn_scope_apart,  $precision_scope_apart_b, $recall_scope_apart, $f1_scope_apart_b);
printf ("Negated B (no cue match): %8d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $negated_g, $negated_p, $tp_negated_apart, $fp_negated_apart, $fn_negated_apart,  $precision_negated_apart_b, $recall_negated_apart, $f1_negated_apart_b);
printf ("Full negation B: %17d | %6d | %4d | %4d | %4d | %13s | %10s | %7s\n",  $cues_g, $cues_p, $tp_full_negation, $fp_full_negation, $fn_full_negation,  $precision_full_negation_b, $recall_full_negation, $f1_full_negation_b);
print "---------------------------+------+--------+------+------+------+---------------+------------+---------\n";
print " # sentences: $count_sentences\n";
print " # negation sentences: $count_sentences_negation\n";
print " # negation sentences with errors: $count_error_sentences_negation\n";
#print " % sentences with errors: $perc_error_sentences\n";
print " % correct sentences: $perc_correct_sentences\n";
print " % correct negation sentences: $perc_correct_negation_sentences\n";

print "--------------------------------------------------------------------------------------------------------\n"; 



close(GOLD);
close(SYSTEM);

