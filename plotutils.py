# Plotting-related code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# Nicer-looking logarithmic axis labeling

def niceLogFunc( x_value, pos ):
	return ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x_value),0)))).format(x_value)	
NiceLogFormatter = ticker.FuncFormatter(niceLogFunc)

def MakeNiceLogAxes( whichAxis="xy", axisObj=None ):
	"""
	Makes one or more axes of a figure display tick labels using non-scientific
	notation (e.g., "0.01" instead of "10^{-2}")
	"""
	
	if axisObj is None:
		ax = plt.gca()
	else:
		ax = axisObj
	if whichAxis in ["x", "xy"]:
		ax.xaxis.set_major_formatter(NiceLogFormatter)
	if whichAxis in ["y", "xy"]:
		ax.yaxis.set_major_formatter(NiceLogFormatter)



def ExtractCenteredSubimage( imData, xc, yc, pix, width=None, height=None, verbose=False ):
    """Extracts and returns a subimage centered at xc,yc, along with 
    corresponding x and y position vectors. If width is None, then the full
    image is returned, along with the corresponding pixel vectors.
    
    Parameters
    ----------
    imData : 2D ndarray of int or float
        the input image
    
    xc, yc : float
        image pixel location (1-based coords) to center extraction around
    
    pix : float
        pixel scale (e.g., arcsec/pix)
    
    width : int or None, optional
        width of subimage to extract, in pixels
        if None, then the entire image is returned (along with xPos, yPos)
    
    height : int or None, optional
        height of subimage to extract, in pixels
        if None, then height = width

    verbose : bool, optional
    
    Returns
    -------
    (imdata_ext, xPos, yPos) : tuple of (2D ndarray, 1D ndarray, 1D ndarray; all of float)
        imData_ext = extracted subimage centered on xc,yc (or entire image if
            width = None)
        xPos = array of pixel coordinate values for x-axis, relative to xc
            e.g., [-1.0, 0.0, 1.0] for 3x3 image centered at 1,1 with pix=1
                  [-0.2, 0.0, 0.2] for 3x3 image centered at 1,1 with pix=0.2
        xPos = array of pixel coordinate values for y-axis, relative to yc
    """

    ySize, xSize = imData.shape
    xPos = pix*(np.arange(1.0, xSize + 1.0) - xc)
    yPos = pix*(np.arange(1.0, ySize + 1.0) - yc)

    if width is not None:
        if height is None:
            height = width
        halfwidth = int(0.5*width)
        x1 = int(xc - halfwidth - 1)
        if (x1 < 0):
            x1 = 0
        x2 = int(xc + halfwidth)
        if (x2 > xSize):
            x2 = -1
        halfheight = int(0.5*height)
        y1 = int(yc - halfheight - 1)
        if (y1 < 0):
            y1 = 0
        y2 = int(yc + halfheight)
        if (y2 > ySize):
            y2 = -1
        xPos = xPos[x1:x2]
        yPos = yPos[y1:y2]
        imdata_ext = imData[y1:y2,x1:x2]
        if verbose:
            print("   pu.ExtractCenteredSubimage: extracting imData[y1:y2,x1:x2] = imData[%d:%d,%d:%d]" % (y1,y2,x1,x2))
    else:
        imdata_ext = imData
    
    return (imdata_ext, xPos, yPos)


def nicecont( imageData, xc=0, yc=0, width=None, height=None, levels=None, pix=1.0,
            axisLabel="pixels", title=None, imageExt=0, log=False, offset=0.0, axesObj=None,
            labelSize=12, labelpad=5, printAxisLabels="both", noErase=False, extraLevels=None,
            color='k', extraColor='r', linewidth=0.5, linestyle='-', secondScale=None,
            secondLabel='kpc', verbose=False ):
    """
    Function which contour-plots an image.
    
    Parameters
    ----------
    imageData : 2D ndarray or str
        2D Numpy array OR FITS image filename (image data is assumed to be
        in 0th header-data unit, unless imageExt is set to something else)
    
    xc, yc : int
        optional center for axes (e.g., center of galaxy) -- by default, these
        are assumed to be IRAF-style 1-based coordinates!
    
    width, height : int
        width and height of subimage (centered on xc,yc) to be plotted;
        if height=None, then a square subimage of size width x width will be extracted
        
    levels : sequence (tuple, list, or Numpy array) of float or None, optional
        contour intensity levels to be plotted (if log=True, then these should be 
        log10 of the original values)
    
    pix : float, optional
        pixel scale (e.g., arcsec/pix or kpc/pix, for axis labeling)
    
    axisLabel : str, optional
        label for x and y axes
    
    title : str, optional
        title for plot
    
    imageExt = int or str, optional
        specification of a different header-data unit in input FITS image 
        (if imageData points to a file)
    
    log : bool, optional
        if True, convert image data to log10(data)

    offset : float, optional
        additive offset to be applied to data (*after* taking log10, if requested)

    axesObj : instance of matplotlib.axes.Axes, optional
        Axes instance to receive the plotting commands

    labelSize : float, optional
        font sizes of x- and y-axis labels
    
    labelpad : float, optional
        shifts position of axis label relative to axis [default=5]

    printAxisLabels : str, optional
        ["both" or "xy", "x", "y"] -- specifies which, if any, of the
        x- or y-axis labels to print

    extraLevels = a list of one or more contour intensity levels to overplot in a
        different color

    color = color for the contours
    
    extraColor = color for contours specified by extraLeveles

    noErase = set this equal to True to draw the contours into an existing plot
        window without erase things first (only used if axesObj is None)

    linewidth = float
    
    linestyle = one of 'solid', 'dashed', 'dashdot', 'dotted'
    
    secondScale = float
        If set, then a second axis scale is drawn (e.g., for pc or kpc)
        value = conversion from original scale (e.g., kpc/arcsec)
    
    secondLabel = str [default = 'kpc']
        label for axis with second scale

    Example:
        >>> nicecont("image.fits", xc=202.4, yc=500.72, levels=np.arange(1.0, 20.0, 0.5))
    """

    # handle case of user supplying a FITS filename
    if type(imageData) == str:
        hdulist = fits.open(imageData)
        imData = hdulist[imageExt].data
    else:
        imData = imageData

    if log is True:
        imData = np.log10(imData)
    imData = imData + offset

    # determine xPos,yPos and extract centered subimage, if requested
    (imData, xPos, yPos) = ExtractCenteredSubimage(imData, xc, yc, pix, width, height,
                                                    verbose=verbose)

    if axesObj is None:
        if noErase is False:
            plt.clf()
        if levels is not None:
            plt.contour(xPos, yPos, imData, levels, colors=color, linewidths=linewidth,
                        linestyles=linestyle)
        else:
            plt.contour(xPos, yPos, imData, colors=color, linewidths=linewidth,
                        linestyles=linestyle)
        if extraLevels is not None:
            plt.contour(xPos, yPos, imData, extraLevels, colors=extraColor, linewidths=1.0,
                        linestyles=linestyle)
        plt.gca().set_aspect('equal')
        if axisLabel is not None:
            if printAxisLabels in ["both", "xy", "x"]:
                plt.xlabel(axisLabel, fontsize=labelSize)
            if printAxisLabels in ["both", "xy", "y"]:
                plt.ylabel(axisLabel, fontsize=labelSize)
        if title is not None:
            plt.title(title)

        if secondScale is not None:
            yrange_orig = np.array(plt.ylim())
            yrange_second = yrange_orig * secondScale
            topy = plt.twinx()
            topy.tick_params(axis='y', length=10)
            topy.tick_params(axis='y', length=5, which="minor")
            topy.set_ylim(yrange_second[0], yrange_second[1])
            plt.ylabel(secondLabel, fontsize=labelSize)
            plt.show()

    else:    # user supplied a matplotlib.axes.Axes object to receive the plotting commands
        if levels is not None:
            axesObj.contour(xPos, yPos, imData, levels, colors=color, linewidths=linewidth,
                            linestyles=linestyle)
        else:
            axesObj.contour(xPos, yPos, imData, colors=color, linewidths=linewidth,
                            linestyles=linestyle)
        if extraLevels is not None:
            plt.contour(xPos, yPos, imData, extraLevels, colors=extraColor, linewidths=0.75,
                        linestyles=linestyle)
        axesObj.set_aspect('equal')
        if axisLabel is not None:
            if printAxisLabels in ["both", "xy", "x"]:
                axesObj.set_xlabel(axisLabel, fontsize=labelSize, labelpad=labelpad)
            if printAxisLabels in ["both", "xy", "y"]:
                axesObj.set_ylabel(axisLabel, fontsize=labelSize, labelpad=labelpad)
        if title is not None:
            axesObj.set_title(title)

        if secondScale is not None:
            xrange_orig = np.array(plt.xlim())
            xrange_second = xrange_orig * secondScale
            topx = plt.twiny()
            topx.tick_params(length=10)
            topx.tick_params(length=5, which="minor")
            topx.set_xlim(xrange_second[0], xrange_second[1])
            plt.xlabel(secondLabel, fontsize=labelSize)
            plt.show()
