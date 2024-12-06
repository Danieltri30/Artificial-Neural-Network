This repo contains the full contents of the CAP5771 Final Project.

There is a .ipynb file (jupyter notebook) that we recommend using, since it will have pre-executed results in it.
Some of the neural network tests take quite a while to run, sometimes 17-22 seconds per epoch for 50 epochs, which is 14.2-18.3.
(it is important to note that this is hardware dependent, since my partner's computer took about 45s per epoch, which means waiting somewhere over half an hour) 
The rest of the script takes approximately a two to three minutes to run (including the first neural network we trained), but of course all of this depends on hardware you posses.

However, for your convenience we have also included a .py version of the project that will display all of the same results.
    Note that the .py file does not include unused code, primarily tokenization of the 'title' and 'body' columns, that ended up not being helpful in the final model.

To run the Main-Recommended.ipynb simply open it in your preferred code editor (such as anaconda's Jupyter notebook) and select run all. 
You may need to install compatible software, such as extensions in visual studio code, in order to run the .ipynb file.
	If you are using visual studio code and want to run the .ipynb, then you will need the Jupyter and Python extensions offered by Microsoft.
	If you have anaconda then you should already have Jupyter Notebooks installed and should be able to open it via the anaconda navigator.

Note that:
	The .ipynb file will already have our results pre-executed, however they can all be re-executed.
    	If you move the ipynb file outside of the folder, then the dataset will need to be moved with it or the code will not work.
    	pip installs for all relevant packages have been included in the .ipynb and can simply be deleted if unwanted or unneeded (they are not included in the .py version).

To run the Main-Alternative.py file open it in your preferred code editor and run it. 
    The text based results will be displayed in console, while the graphs will be displayed in pop-up boxes.
    The script will not proceed until the graph pop-up boxes have been closed.

When either of the scripts is finished a single line stating that "Script Finished." will be displayed in console or at the bottom of the page in the case of the .ipynb file.
