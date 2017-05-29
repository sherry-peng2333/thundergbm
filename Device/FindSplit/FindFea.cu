/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "IndexComputer.h"
#include "FindFeaKernel.h"
#include "../Hashing.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/HostUtility.h"
#include "../../SharedUtility/CudaMacro.h"

using std::cout;
using std::endl;
using std::make_pair;
using std::cerr;

/**
 * @brief: rearrange marker for computing feature length and start pos of each node
 */
__global__ void RearrangeData(const int *pOldInsId, const real *pOldFvalue, const int *pOldFeaId,
							  const unsigned int *pDstIndexEachFeaValue,
							  int numFeaValue, int *pNewInsId, real *pNewFvalue, int *pNewFeaId)
{
	//one thread loads one value
	//## global id looks ok, but need to be careful
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	//index for scatter
	int idx = pDstIndexEachFeaValue[gTid];
	if(idx == -1)//instance is in a leaf node
		return;

	//scatter: store GD, Hess and the feature value.
	pNewInsId[idx] = pOldInsId[gTid];
	pNewFvalue[idx] = pOldFvalue[gTid];
	pNewFeaId[idx] = pOldFeaId[gTid];
}

/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat, void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//reset memory for this bag
	{
		manager.MemsetAsync(bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);

		manager.MemsetAsync(bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(double) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute index for each feature value
	IndexComputer indexComp;
	unsigned int *pFeaValueStartPosEachNode_h = new unsigned int[bagManager.m_maxNumSplittable];

	KernelConf conf;
	int blockSizeLoadGD;
	dim3 dimNumofBlockToLoadGD;
	conf.ConfKernel(indexComp.m_totalFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
	if(numofSNode > 1)
	{
		clock_t comIdx_start = clock();
		//compute gather index via GPUs
		indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, bagId);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);

		unsigned int *pTmpFvalueStartPosEachNode = bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
		checkCudaErrors(cudaMemcpy(pFeaValueStartPosEachNode_h, pTmpFvalueStartPosEachNode,
								   sizeof(unsigned int) * bagManager.m_maxNumSplittable, cudaMemcpyDeviceToHost));
	
		//copy # of feature values of each node
		manager.MemcpyHostToDeviceAsync(indexComp.m_pNumFeaValueEachNode_dh, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(unsigned int) * bagManager.m_maxNumSplittable, pStream);
		//copy feature value start position of each node
		//copy (in pinned mem) of feature values for each feature in each node
		manager.MemcpyHostToDeviceAsync(bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										bagManager.m_pEachFeaLenEachNodeEachBag_dh + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(int) * bagManager.m_maxNumSplittable * bagManager.m_numFea, pStream);
	
		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
		clock_t start_gd = clock();
		//scatter operation
		//total fvalue to load may be smaller than m_totalFeaValue, due to some nodes becoming leaves.
		int numFvToLoad = pFeaValueStartPosEachNode_h[numofSNode - 1] + indexComp.m_pNumFeaValueEachNode_dh[numofSNode - 1];
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_numIns, manager.m_pDInsId, manager.m_pdDFeaValue,
															   bagManager.m_pIndicesEachBag_d, numFvToLoad,
															   bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   	bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   	   	manager.m_pDInsId, manager.m_pdDFeaValue, indexComp.m_totalFeaValue,
															   		bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		//copy # of feature values of a node
		manager.MemcpyHostToDeviceAsync(&manager.m_numFeaValue, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(unsigned int), pStream);
		//copy feature value start position of each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 	 sizeof(unsigned int), pStream);
		//copy each feature start position in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(unsigned int) * nNumofFeature, pStream);
		//copy # of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									    sizeof(int) * nNumofFeature, pStream);

		//copy # (in pinned mem) of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_dh + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(int) * nNumofFeature, pStream);

		//set indexComp
		pFeaValueStartPosEachNode_h[0] = 0;
		indexComp.m_pNumFeaValueEachNode_dh[0] = manager.m_numFeaValue;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
	}

//	cout << "prefix sum" << endl;
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	int totalNumArray = indexComp.m_numFea * numofSNode;
	cudaStreamSynchronize((*(cudaStream_t*)pStream));//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled

	//construct keys for exclusive scan
	int *pnKey_d;
	int keyFlag = 0;
	unsigned int *pnLastFvalueOfThisFvalue_d, *pnLastFvalueOfThisFvalue_h = new unsigned int[bagManager.m_numFeaValue];
	checkCudaErrors(cudaMalloc((void**)&pnKey_d, bagManager.m_numFeaValue * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&pnLastFvalueOfThisFvalue_d, bagManager.m_numFeaValue * sizeof(unsigned int)));
	unsigned int *pTempEachFeaStartEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	unsigned int *pTempEachFeaStartEachNode_h = new unsigned int[totalNumArray];
	checkCudaErrors(cudaMemcpy(pTempEachFeaStartEachNode_h, pTempEachFeaStartEachNode, sizeof(unsigned int) * totalNumArray, cudaMemcpyDeviceToHost));
	for(int m = 0; m < totalNumArray; m++){
		unsigned int arrayLen = bagManager.m_pEachFeaLenEachNodeEachBag_dh[m];
		unsigned int arrayStartPos = pTempEachFeaStartEachNode_h[m];
		if(arrayLen == 0){
			if(m == totalNumArray - 1)
				continue;
			if(arrayStartPos != pTempEachFeaStartEachNode_h[m + 1])
				printf("%u v.s. %u, m=%d, totalNumArray=%d\n", arrayStartPos, pTempEachFeaStartEachNode_h[m + 1], m,
						totalNumArray);
			PROCESS_ERROR(arrayStartPos == pTempEachFeaStartEachNode_h[m + 1]);
			continue;
		}
//		printf("start=%d, len=%d, m=%d, totalNumArray=%d\n", arrayStartPos, arrayLen, m, totalNumArray);
		checkCudaErrors(cudaMemset(pnKey_d + arrayStartPos, keyFlag, sizeof(int) * arrayLen));
		if(keyFlag == 0)
			keyFlag = -1;
		else 
			keyFlag = 0;
		//assign fealen to each fvalue
		for(int fv = 0; fv < arrayLen; fv++){
			pnLastFvalueOfThisFvalue_h[fv + arrayStartPos] = arrayStartPos + arrayLen - 1;
		}
	}
	checkCudaErrors(cudaMemcpy(pnLastFvalueOfThisFvalue_d, pnLastFvalueOfThisFvalue_h, sizeof(unsigned int) * bagManager.m_numFeaValue, cudaMemcpyHostToDevice));

	//compute prefix sum for gd and hess (more than one arrays)
	double *pTempGDSum = bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	real *pTempHessSum = bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey_d, pnKey_d + bagManager.m_numFeaValue, pTempGDSum, pTempGDSum);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey_d, pnKey_d + bagManager.m_numFeaValue, pTempHessSum, pTempHessSum);

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	checkCudaErrors(cudaFree(pnKey_d));
	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//default to left or right
	bool *pDefault2Right;
	checkCudaErrors(cudaMalloc((void**)&pDefault2Right, sizeof(bool) * bagManager.m_numFeaValue));
	checkCudaErrors(cudaMemset(pDefault2Right, 0, sizeof(bool) * bagManager.m_numFeaValue));

	//cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	//# of feature values that need to compute gains; the code below cannot be replaced by indexComp.m_totalNumFeaValue, due to some nodes becoming leaves.
	int numofDenseValue = pFeaValueStartPosEachNode_h[numofSNode - 1] + indexComp.m_pNumFeaValueEachNode_dh[numofSNode - 1];

	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(numofDenseValue, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
											numofSNode,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
											numofDenseValue, pnLastFvalueOfThisFvalue_d,
											bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
											pDefault2Right);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");
	
	delete[] pnLastFvalueOfThisFvalue_h;
	cudaDeviceSynchronize();

	//for testing gain before fixing
	real *pGain = new real[bagManager.m_numFeaValue];
	cudaMemcpy(pGain, bagManager.m_pGainEachFvalueEachBag, bagManager.m_numFeaValue * sizeof(real), cudaMemcpyDeviceToHost);

	//change the gain of the first feature value to 0
	int numFeaStartPos = bagManager.m_numFea * numofSNode;
//	printf("num fea start pos=%d (%d * %d)\n", numFeaStartPos, bagManager.m_numFea, numofSNode);
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numFeaStartPos, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
																bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
																numFeaStartPos, bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
																bagManager.m_numFeaValue);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after FirstFeaGain");

	clock_t end_comp_gain = clock();
	total_com_gain_t += (end_comp_gain - start_comp_gain);

//	cout << "searching" << endl;
	clock_t start_search = clock();
	//compute # of blocks for each node
	unsigned int maxNumFeaValueOneNode = 0;
	unsigned int testTotalFeaValue = 0;
	for(int n = 0; n < numofSNode; n++)
	{//find the node with the max number of element
		if(maxNumFeaValueOneNode < indexComp.m_pNumFeaValueEachNode_dh[n])
			maxNumFeaValueOneNode = indexComp.m_pNumFeaValueEachNode_dh[n];
		testTotalFeaValue += indexComp.m_pNumFeaValueEachNode_dh[n];
//		printf("fv start=%u, len=%d\t", pFeaValueStartPosEachNode_h[n], indexComp.m_pNumFeaValueEachNode_dh[n]);
	}
//	printf("monitored total fvalue: %u\n", testTotalFeaValue);

	delete []pFeaValueStartPosEachNode_h;
	delete[] pTempEachFeaStartEachNode_h;

	PROCESS_ERROR(maxNumFeaValueOneNode > 0);
	int blockSizeLocalBestGain;
	dim3 dimNumofBlockLocalBestGain;
	conf.ConfKernel(maxNumFeaValueOneNode, blockSizeLocalBestGain, dimNumofBlockLocalBestGain);
	PROCESS_ERROR(dimNumofBlockLocalBestGain.z == 1);
	dimNumofBlockLocalBestGain.z = numofSNode;//each node per super block
	int numBlockPerNode = dimNumofBlockLocalBestGain.x * dimNumofBlockLocalBestGain.y;
	//find the block level best gain for each node
	PickLocalBestSplitEachNode<<<dimNumofBlockLocalBestGain, blockSizeLocalBestGain, 0, (*(cudaStream_t*)pStream)>>>(
								bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
								bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
								bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after PickLocalBestSplitEachNode");

	//find the global best gain for each node
	if(numBlockPerNode > 1){
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerNode, blockSizeBestGain, dimNumofBlockDummy);
		PickGlobalBestSplitEachNode<<<numofSNode, blockSizeBestGain, 0, (*(cudaStream_t*)pStream)>>>(
									bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
									bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
									bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
									bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
								    numBlockPerNode, numofSNode);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		GETERROR("after PickGlobalBestSplitEachNode");
	}
	else{//local best fea is the global best fea
		manager.MemcpyDeviceToDeviceAsync(bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
										bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(real) * numofSNode, pStream);
		manager.MemcpyDeviceToDeviceAsync(bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
											sizeof(int) * numofSNode, pStream);
	}

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t end_search = clock();
	total_search_t += end_search - start_search;

	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
									 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
									 bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
				  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 pDefault2Right, pnLastFvalueOfThisFvalue_d,
				  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	cudaFree(pnLastFvalueOfThisFvalue_d);
	cudaFree(pDefault2Right);
}
