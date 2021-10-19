from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting



class strehlCalc:   

        
# ###### Current attributes:

# User inputs upon object creation: ------------------------

# All data should be .fits files
    
# All paths must be in a list!
    # ex) 
        #pathToData=[None]*N
        #pathToData[0]='path'
    
#     pathToData:     Path where the image data of the closed loop PSFs are stored. Each .fits file should have one data cube of image frames.

#     pathToFlats:    path where the PSF response to a flat wavefront is stored.                      
#                     This is what the closed loop PSFs are compared to in the Strehl calculation

#     pathToDarks:    If you have a dark image, input the path.
#                     Can be a single frame (Hopefully a mean frame from multiple dark frames), 
#                     or a data cube of individual dark frames that will be averaged for you.
#                     If not, the background will be estimated from the image frame.

#     drkSubtractDiameter:     
#                     Diameter in pixels of the circular mask that masks out the PSF 
#                     for the dark subtraction that is estimated from the image frame. 
#                     * A common fix to a bad strehl measurement is decreasing the diameter size. 
#                     If none is given the diameter of the mask will be 1/3 the smallest dimension of the data frame . 

# Class generated attributes: --------------------------

## Saturation check. Each index corresponds to the file opened in the list pathTo... with the same index.

### The saturated value for the Basler Cameras is 1023, and is a hard coded value. If your camera has a different saturation value change it in code. 

#     darkSat:        True: dark data is oversaturated. False: dark data is not saturated
#     imageSat:       True: there are one or more frames in that file that are oversaturated. False: no frames in that file are oversaturated.
#     perfSat:        True: there are one or more frames in that file that are oversaturated. False: no frames in that file are oversaturated. 


#     imagePeak:      The peak values of the PSF used in Strehl Calculation
#     perfPeak:       The peak values of the perfect PSF the closed loop PSFs will be compared to in the Strehl Calculation
#     imgFlux:        The sum of the flux values of the PSF used in the Strehl Calculation. The flux is summed over circles of increasing diameter. 
#     perfFlux:       The sum of the flux values of the perfect PSF the closed loop PSFs will be compared to in the Strehl Calculation. The flux is summed over circles of increasing diameter.
#     fluxDiameter:   The diameters in pixels that the flux was summed over.

#     imageSize:      Size of the data cube containing the closed loop PSF data
#     perfSize:       Size of the data cube containing the images of the flat wavefront response PSF data
#     darkSize:       Size of the data cube containing the dark frames.

#     self.Strehl:    The calculated Strehl ratio for each image file, at each 

### If smart peak finding is chosen: ---------------------
#     FWHM:           The Full Width Half Max of the PSF core in pixels given by either user input, or estimated in the pipeline.
#     radialA:        The radial average computed by the pipeline if the FWHM needs to be estimated
#     totalProfile:   The total profile of the PSF, computed by mirroring and combining the radial average with its self to give a full 2D span.
#
#


        

    
    def __init__(self, pathToData,pathToFlats, pathToDarks=None,FWHM=None, smartPeak=None,drkSubtractDiameter=None):
        self.pathToData=pathToData
        self.pathToFlats=pathToFlats
        if pathToDarks:
            self.pathToDarks=pathToDarks
            self.darkSat, self.darkData, self.darkSize=self.openFile(self.pathToDarks)
        else:
            pathToDarks=None
            self.darkSat=False 
            
        #open the files, test if any are oversaturated  
        self.imageSat, imageData, self.imageSize=self.openFile(self.pathToData)
        self.perfSat, perfData, self.perfSize=self.openFile(self.pathToFlats)
            
        ## Check to make sure the data is not oversaturated and compatible with the pipeline. 
        self.errorMsg()
        
        if smartPeak:
            if FWHM:
               self.FWHM=FWHM 
            else:
                print("Estimating the FWHM.")
                self.radialA, self.totalProfile=self.radialAveragePrep(perfData)
                self.FWHM=self.fwhmEstimate()
            print('Finding the peak of the data using DAOStarFinder.')
            xcentroid, ycentroid=self.smartPeakFinder(imageData)
            xPerfcentroid, yPerfcentroid=self.smartPeakFinder(perfData)
                
        else:
            print('Finding the peak using the max value in data.')
            xcentroid, ycentroid=self.peakFinder(imageData) ### RETURN ONLY THE CENTROIDS
            xPerfcentroid, yPerfcentroid=self.peakFinder(perfData)
            
        
        # dark subtract + pull out image peak
        pData=self.darkSubtract(imageData,xcentroid, ycentroid)
        pPerfData=self.darkSubtract(perfData,xPerfcentroid, yPerfcentroid)
        print("Finished dark subtraction.")
        
        #Assumes the coordinates of the peak haven't changed after dark subtraction
        
        self.imagePeak=self.coord2peak(pData, xcentroid, ycentroid)
        self.perfPeak=self.coord2peak(pPerfData, xPerfcentroid, yPerfcentroid)
        print("Found the peak in each image.")
        
#       #Sum the Flux in a circle centered on the PSF at increasing radii.  
        
        self.fluxDiameters, self.imageFlux=self.sumFlux(xcentroid, ycentroid,pData)
        self.perfFluxDiameter, self.perfFlux=self.sumFlux(xPerfcentroid, yPerfcentroid, pPerfData)
        print("Summed the flux in each image.")
        
#         # Calculate the Strehl value. 
        print("Calculating the Strehl Ratio.")
        self.Strehl=self.strehlCalculator()
        print("Finished.")
        
    
    
##### Functions -------------------------------------------------------------
        
                      
    def openFile(self, path):
            
        #check if it is a single file or multiple
        fileSize=np.shape(path)
        
        ## Pull a test file
        testFile=fits.open(path[0])
        testData=testFile[0].data
        dataSize=np.shape(testData)
        
        data=np.zeros((*fileSize,*dataSize))
        oversaturated=fileSize[0]*[None]
        
        data[0,:]=testData
        oversaturated[0]=np.any(testData == 1023)
        
        # Pull out the rest of the data and check if it is oversaturated
        for i in range(fileSize[0]-1):
            i=i+1
            testFile=fits.open(path[i])
            data[i,:]=testFile[0].data
            oversaturated[i]=np.any(data[i,:] == 1023)
        
        imageSize=np.shape(data)

            
        return oversaturated, data, imageSize
                           
    
    def darkSubtract(self, data,xcentroid, ycentroid, drkSubtractDiameter=None):
        
        imageSize=np.shape(data)
        
        #takes the mean if it is a data cube
        if hasattr(self, 'darkData'):
            dkData=np.mean(self.darkData,axis=0)
            for i in range(imageSize[0]):
                data[i,:]=np.subtract(data[i,:],dkData)
            print("Subtracted Dark Frame")

        
        ## Find peak of data to center mask: Assumes the max value in image is the peak.

#         xcentroid=np.zeros((imageSize[0],imageSize[1]))
#         ycentroid=np.zeros((imageSize[0],imageSize[1]))
        
#         for i in range(imageSize[0]):
#             for j in range(imageSize[1]):
#                     max_xy=np.where(data[i,j,:,:]==data[i,j,:,:].max())
#                     xcentroid[i,j]=max_xy[1][0]
#                     ycentroid[i,j]=max_xy[0][0]
        
                
        ### Second round of background subtraction based of the image background.
        
        # Places a circle mask over the PSF data of diameter given by user input. 
        #If no user input, it uses a diameter=1/3 smallest dimension of a single data frame. 
        #After, row by row takes the mean value of the row, and subtracts it from that row. 
        
            
        if drkSubtractDiameter == None:
            check=min(imageSize[2],imageSize[3])
            drkSubtractDiameter=np.round(check/3)
        
        pData=np.zeros(imageSize)
        
        for i in range(imageSize[0]):
            for j in range(imageSize[1]):
                mask=self.circle(imageSize[2],imageSize[3],drkSubtractDiameter,xcentroid[i,j],ycentroid[i,j])
                mask=1-mask
                dummy=mask*data[i,j,:]
                for k in range(imageSize[2]):
                    m=np.mean(dummy[k,:])
                    pData[i,j,k,:]=np.subtract(data[i,j,k,:],m)
                    

            
        return pData
    
    def coord2peak(self, data, xcentroid, ycentroid):
        imageSize=np.shape(data)
        peak=np.zeros((imageSize[0],imageSize[1]))

        for i in range(imageSize[0]):
             for j in range(imageSize[1]):
                    peak[i,j]=data[i,j,int(ycentroid[i,j]),int(xcentroid[i,j])]
        
        return peak
    
    
    def peakFinder(self,data):
    #Stupid Peak finder. Pulls out the max value of the image as the peak. 
    ## Pull out the values of the PSF peak after dark subtraction
        imageSize=np.shape(data)
        xcentroid=np.zeros((imageSize[0],imageSize[1]))
        ycentroid=np.zeros((imageSize[0],imageSize[1]))
        
        
        for i in range(imageSize[0]):
            for j in range(imageSize[1]):
                max_xy=np.where(data[i,j,:,:]==data[i,j,:,:].max())
                xcentroid[i,j]=max_xy[1][0]
                ycentroid[i,j]=max_xy[0][0]
            
        return xcentroid, ycentroid
        
        
    def smartPeakFinder(self,data):
        #Smart Peak finder using photutils from the astropy package
        fwhmSize=np.shape(self.FWHM)
        imageSize=np.shape(data)
        xcentroid=np.zeros((imageSize[0],imageSize[1]))
        ycentroid=np.zeros((imageSize[0],imageSize[1]))
        
        if fwhmSize[0]==1:

            for i in range(imageSize[0]):
                sdev=np.std(data[i,:])
                daofind = DAOStarFinder(fwhm=self.FWHM[0], threshold=5.*sdev)
                for j in range(imageSize[1]):
                    sources=daofind(data[i,j,:])
                    for col in sources.colnames:
                        sources[col].info.format = '%.8g'
                        peak=max((sources["peak"]))
                        ind=np.where(sources["peak"]==max(sources["peak"]))
                        xc=(sources["xcentroid"][ind[0]])
                        yc=(sources["ycentroid"][ind[0]])     
                        xcentroid[i,j]=int(np.round(xc[0]))
                        ycentroid[i,j]=int(np.round(yc[0]))
        else:  
            for i in range(imageSize[0]):
                sdev=np.std(data[i,:])
                daofind = DAOStarFinder(fwhm=self.FWHM[i], threshold=5.*sdev)
                for j in range(imageSize[1]):
                    sources=daofind(data[i,j,:])
                    for col in sources.colnames:
                        sources[col].info.format = '%.8g'
                        peak=max((sources["peak"]))
                        ind=np.where(sources["peak"]==max(sources["peak"]))
                        xc=(sources["xcentroid"][ind[0]])
                        yc=(sources["ycentroid"][ind[0]])     
                        xcentroid[i,j]=int(np.round(xc[0]))
                        ycentroid[i,j]=int(np.round(yc[0]))
                        
        return xcentroid, ycentroid


    def fwhmEstimate(self):
        from astropy.modeling import models, fitting 
        s=np.round(len(self.totalProfile[0,:])/2)
        x=np.arange(-s+1,s,1)
        
        sz=np.shape(self.totalProfile)
        FWHM=np.zeros(sz[0])
        FWHMval=np.zeros(sz[0])
        
        #fit Guassian to data
        for i in range(sz[0]):
            g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, x, self.totalProfile[i,:])
            
            gaus=g(x)
            stdv=np.std(gaus)
            FWHMval[i]=2*np.sqrt(2*np.log(2))*stdv
            
            val=gaus[gaus>=FWHMval[i]]
            FWHM[i]=len(val)
            
        return FWHM
            
            
    
    def radialAveragePrep(self,data):
        
        #calculates an estimated radial average in one frame of data per file. 
        #This is meant to be passed to the Full width half max estimator.
        
        imageSize=np.shape(data)
        xcentroid=np.zeros((imageSize[0]))
        ycentroid=np.zeros((imageSize[0]))
        
        for i in range(imageSize[0]):
            max_xy=np.where(data[i,0,:,:]==data[i,0,:,:].max())
            xcentroid[i]=max_xy[1][0]
            ycentroid[i]=max_xy[0][0]
        
        
        if imageSize[2] != imageSize[3]:
            check=min(imageSize[2],imageSize[3])
            radData=np.zeros(imageSize[0],1,check, check)
            for i in range(imageSize[0]):
                radData[i,0,:]=self.cropper(data[i,0,:,:],xcentroid[i], ycentroid[i], check)
        else:
            radData=np.zeros((imageSize[0],1,imageSize[2],imageSize[3]))
            for i in range(imageSize[0]):
                radData[i,0,:,:]=data[i,0,:,:]
                
        
        test=self.radial_data_median_only(radData[0,0,:,:],1)
        radSize=np.shape(test)
        radialA=np.zeros((imageSize[0],radSize[0]))
        totalProfile=np.zeros((imageSize[0],radSize[0]*2-1))
        for i in range(imageSize[0]):
            radialA[i,:]=self.radial_data_median_only(radData[i,0,:,:],1)
#             m=np.mean(radialA[i,0:100])
#             radialA[i,:]=radialA[i,:]-m
            flip=np.flip(radialA[i,:])
            totalProfile[i,:]=np.concatenate((flip, radialA[i,1:]))
        
        return radialA, totalProfile
        
  
        
    
    def sumFlux(self, xcentroid, ycentroid, data):
        
        
        imageSize=np.shape(data)
        check=min(imageSize[2],imageSize[3])
        x=np.floor(np.floor(((np.floor((check/10))*10)))/10)*10-50
        r1=range(20, int(x), 2)
        sz2=np.shape(r1)
        rFlux=np.zeros((imageSize[0],imageSize[1],sz2[0]))
        
        for j in range(imageSize[0]):
            count=0
            for r in r1:
                for i in range(imageSize[1]):
                    mask=self.circle(imageSize[2],imageSize[3], r, xcentroid[j,i], ycentroid[j,i])
                    mData=data[j,int(i),:,:]*mask
                    rFlux[j,int(i),count]=np.sum(mData)
                count=count+1 
        
        return r1, rFlux   
        
    def strehlCalculator(self):   
        #if only one data cube is given use it for all frames
        #if the same number of data cubes are given match the PSFs to the same index as the corresponding Flat
        
        # if there is a corresponding flat wavefront PSF data cube to each of the closed loop PSF data cubes
        
        if self.perfSize[0]>1:
            
            sz=np.shape(self.imageFlux)
            Strehl=np.zeros((sz[0],sz[2]))
            for i in range(sz[0]): 
                for j in range(sz[2]):  
                    d=np.mean(self.imagePeak[i,:]/self.imageFlux[i,:,j])
                    p=np.mean((self.perfPeak[i,:]/self.perfFlux[i,:,j]))
                    Strehl[i,j]=d/p
                    
        
        # if there is a single flat wavefront PSF data cube corresponding to all of the closed loop PSF data cubes
        if self.perfSize[0]==1:
            print("I am using the same reference PSF to calculate Strehl for all data sets.")
            sz=np.shape(self.imageFlux)
            Strehl=np.zeros((sz[0],sz[2]))

            for i in range(sz[0]): 
                for j in range(sz[2]):  
                    d=np.mean(self.imagePeak[i,:]/self.imageFlux[i,:,j])
                    p=np.mean(self.perfPeak[0,:]/self.perfFlux[0,:,j])
                    Strehl[i,j]=d/p
        
        return Strehl
        
    def circle(self,s,v,d,x,y):
        s=s+1
        v=v+1
        cen=np.array([x,y])
        r=np.zeros([s,v])
        a=np.arange(-cen[0]+1,v-cen[0],1)
        b=np.arange(-cen[1]+1,s-cen[1],1)
        [X,Y]=np.meshgrid(a,b)
        r=np.sqrt(np.double((X**2+Y**2)))
        c = (r<=d/2).astype(int)
        return c
        
        
    def errorMsg(self):
        
        if self.perfSize[0]>1 and self.perfSize[0] != self.imageSize[0]:
            print("Error: The flat wavefront PSF data frames must be either a single data cube, or a number of data cubes equalling the number of closed loop PSF data cubes.")
            raise ValueError
        if self.imageSat == True:
            print("One or more of the closed loop PSF data frames is oversaturated.")
            raise ValueError
        if self.perfSat == True:
            print("One or more of the flat wavefront PSF data frames is oversaturated.")
            raise ValueError
        if self.darkSat == True:
            print("One of more of the dark frames is oversaturated.")
            raise ValueError
                           
    def radial_data_median_only(self,data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    #Pared down version of radial_data that computes only the median radial profile
        data = np.array(data)
    
        if working_mask==None:
            working_mask = np.ones(data.shape,bool)
    
        npix, npiy = data.shape
        if x==None or y==None:
            x1 = np.arange(-npix/2.,npix/2.)
            y1 = np.arange(-npiy/2.,npiy/2.)
            x,y = np.meshgrid(y1,x1)

        r = abs(x+1j*y)

        if rmax==None:
            rmax = r[working_mask].max()

        dr = np.abs([x[0,0] - x[0,1]]) * annulus_width
        radial = np.arange(rmax/dr)*dr + dr/2.
        nrad = len(radial)
        #radialdata = radialDat()
        radialdata_median = np.zeros(nrad)

        for irad in range(nrad): #= 1:numel(radial)
            minrad = irad*dr
            maxrad = minrad + dr
            thisindex = (r>=minrad) * (r<maxrad) * working_mask
            if not thisindex.ravel().any():
                radialdata_median[irad] = np.nan
            else:
                radialdata_median[irad] = np.nanmedian(data[thisindex])
    
        return radialdata_median
                           
    def cropper(self,data, x, y, d):
    ## takes the imput matrix (data) and crops a dxd square around the coordinate (x,y). Returns the cropped image

        box=pData[y-d:y+d,x-d:x+d]
        return box

        
    
    
        
        
            