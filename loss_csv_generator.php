<!doctype html>
<html>
<head>
<meta charset="UTF-8">
<title>Data CSV Generator</title>

<style>
	body {
		font-family: "Lucida Grande", "Lucida Sans Unicode", "Lucida Sans", "DejaVu Sans", Verdana, "sans-serif";
	}
	#main {
		width: 80vw;
		margin-left: auto;
		margin-right: auto;
	}
	h1 {
		margin-bottom: 0;
	}
	h3 {
		margin-top: 0;
	}
	blockquote {
		font-family: Courier New;
	}
	.error {
		color: red;
		font-weight: bold;
	}
	.success {
		color: green;
		font-weight: bold;
	}
</style>

</head>

<?php
	if (isset($_POST["submit"])) {
		$name = $_POST["dataFile"];
		$extension = end((explode(".", $name)));
		$csvExtension = end((explode(".", $_POST["fileName"])));
		
		if ($csvExtension == "csv") {
			if ($extension == "txt") {
				$data = file($name);
				if (empty($data)) {
					$error = "File either does not exist or has no data.";
				} else {
					$stepData = array();

					for ($i = 0; $i < count($data); $i++) {
						$first = substr($data[$i], 0, 4);

						if ($first != "Step") {
							continue;
						}

						$dataPoint = array();

						$stripStep = substr($data[$i], 5);
						$colon = strpos($stripStep, ":");

						if ($_POST["oneIndex"]) {
							$dataPoint["step"] = substr($stripStep, 0, $colon) + 1;
						} else {
							$dataPoint["step"] = substr($stripStep, 0, $colon);
						}
						$dataPoint["loss"] = substr($stripStep, $colon + 9, 9);

						$stepData[] = $dataPoint;
					}

					// create a file pointer connected to the output stream
					$output = fopen($_POST["fileName"], 'w');

					// output the column headings
					fputcsv($output, array('Step', 'Loss'));

					for ($i = 0; $i < count($stepData); $i++) {
						fputcsv($output, $stepData[$i]);
					}

					$complete = 1;
				}
			} else {
				$error = "Not a text file!";
			}
		} else {
			$error = "Not a CSV file!";
		}
	}
?>

<body>
	<div id="main">
		<h1>Data CSV Generator Widget</h1>
		<h3>Created by DG Kim</h3>
		This widget is created to speed up graph creation for the data for the loss from the FFNN. It takes data of the format
		<blockquote>Step 23: loss = 0.6777992 (1.938 sec)</blockquote> and reduces them to one row with the step information and the loss. (Generation time data is ignored)
		<hr>
		<form action="loss_csv_generator.php" method="POST" enctype="multipart/form-data">
			<?php
				if (isset($error)) {
			?>
			<span class="error"><?php echo($error); ?></span><br />
			<?php
				}
			?>
			<?php
				if (isset($complete)) {
			?>
			<span class="success">Data exported to CSV successfully!</span><br />
			<?php
				}
			?>
			Name of the .txt file here (must reside in this directory):<br />
			<input type="text" name="dataFile" id="dataFile"> <br />
			Name of desired .csv file (include extension):<br />
			<input type="text" name="fileName" id="fileName"> <br />
			<input type="checkbox" checked="checked" name="oneIndex"> Output one indexed steps? (Adds 1 to each step)<br />
			<input type="submit" value="Submit" name="submit">
		</form>
	</div>
</body>
</html>