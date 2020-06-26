The program makes use of the Python libraries numpy, matplotlib.
The program needs Gambit to compute Nash equilibria correctly. If not installed, download and install
Gambit 15:
http://www.gambit-project.org/

Copy the gambit-enummixed.exe executable from the installation folder to the folder containing the Nash.py script.

To store the images from the replicator dynamic, create a folder "/img" inside the folder containing the
Main.py script.
The method game_session() in Main.py contains a call to to_latex() of GrandTable.py. This method converts the
resulting grand table of the game into a formatted table suitable for LaTeX. To store the tables, create a
folder "/grand_table" inside the folder containing the Main.py script.

Run Main.py to play the games and compute the statistics.