using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;

namespace Pastasfuture.KDTree.Runtime
{
    public static class KDTree
    {
        public struct KDTreeData
        {
            public NativeArray<int> payloadIndices;
            public NativeArray<float3> positions;
            public NativeArray<float> radii;
            public NativeArray<float3> aabbs;
            public NativeArray<float> splitPositions;
            public NativeArray<byte> splitAxes;

            public static readonly KDTreeData zero = new KDTreeData();
        }

        public struct KDTreeHeader
        {
            public int count;
            public int holeCount;
            public int depthCount;
            public int depthCapacity;

            public static readonly KDTreeHeader zero = new KDTreeHeader()
            {
                count = 0,
                holeCount = 0,
                depthCount = 0,
                depthCapacity = 0
            };
        }

        public static void Dispose(ref KDTreeHeader header, ref KDTreeData data)
        {
            if (data.payloadIndices.IsCreated) { data.payloadIndices.Dispose(); }
            if (data.positions.IsCreated) { data.positions.Dispose(); }
            if (data.radii.IsCreated) { data.radii.Dispose(); }
            if (data.aabbs.IsCreated) { data.aabbs.Dispose(); }
            if (data.splitPositions.IsCreated) { data.splitPositions.Dispose(); }
            if (data.splitAxes.IsCreated) { data.splitAxes.Dispose(); }

            header.count = 0;
            header.holeCount = 0;
            header.depthCount = 0;
            header.depthCapacity = 0;
        }

        [BurstCompile]
        public static int Count(ref KDTreeHeader header, ref KDTreeData data)
        {
            return IsCreated(ref header, ref data) ? header.count : 0;
        }

        [BurstCompile]
        public static int CountValid(ref KDTreeHeader header, ref KDTreeData data)
        {
            return IsCreated(ref header, ref data) ? (header.count - header.holeCount) : 0;
        }

        [BurstCompile]
        public static int Capacity(ref KDTreeHeader header, ref KDTreeData data)
        {
            return IsCreated(ref header, ref data) ? data.positions.Length : 0;
        }

        [BurstCompile]
        public static bool IsCreated(ref KDTreeHeader header, ref KDTreeData data)
        {
            return data.positions.IsCreated;
        }

        public static void Allocate(ref KDTreeHeader header, ref KDTreeData data, int capacity, int depthCapacity, Allocator allocator)
        {
            Debug.Assert(!IsCreated(ref header, ref data));

            data.payloadIndices = new NativeArray<int>(capacity, allocator);
            data.positions = new NativeArray<float3>(capacity, allocator);
            data.radii = new NativeArray<float>(capacity, allocator);
            header.count = 0;
            header.holeCount = 0;

            int boundsCount = 1 << depthCapacity;
            data.aabbs = new NativeArray<float3>(boundsCount * 2, allocator);
            data.splitPositions = new NativeArray<float>(boundsCount >> 1, allocator);
            data.splitAxes = new NativeArray<byte>(boundsCount >> 1, allocator);
            header.depthCapacity = depthCapacity;
            header.depthCount = 0;
        }

        public static void Copy(ref KDTreeHeader headerSrc, ref KDTreeData dataSrc, ref KDTreeHeader headerDst, ref KDTreeData dataDst, int count, int depthCount)
        {
            Debug.Assert(Capacity(ref headerSrc, ref dataSrc) >= count);
            Debug.Assert(Capacity(ref headerDst, ref dataDst) >= count);
            Debug.Assert(headerSrc.count >= count);

            NativeArray<int>.Copy(dataSrc.payloadIndices, dataDst.payloadIndices, count);
            NativeArray<float3>.Copy(dataSrc.positions, dataDst.positions, count);
            NativeArray<float>.Copy(dataSrc.radii, dataDst.radii, count);

            int boundsCount = 1 << depthCount;
            Debug.Assert(depthCount <= headerSrc.depthCapacity);
            Debug.Assert(depthCount <= headerDst.depthCapacity);
            NativeArray<float3>.Copy(dataSrc.aabbs, dataDst.aabbs, boundsCount);
            NativeArray<float>.Copy(dataSrc.splitPositions, dataDst.splitPositions, boundsCount >> 1);
            NativeArray<byte>.Copy(dataSrc.splitAxes, dataDst.splitAxes, boundsCount >> 1);

            headerDst.count = count;
            headerDst.holeCount = headerSrc.holeCount;
            headerDst.depthCount = depthCount;
        }

        public static void EnsureCapacity(ref KDTreeHeader header, ref KDTreeData data, int capacityRequested, int depthCapacityRequested, Allocator allocator)
        {
            if (Capacity(ref header, ref data) >= capacityRequested
                && header.depthCapacity >= depthCapacityRequested) { return; }

            KDTreeData dataNext = new KDTreeData();
            KDTreeHeader headerNext = new KDTreeHeader();
            Allocate(ref headerNext, ref dataNext, capacityRequested, depthCapacityRequested, allocator);
            if (IsCreated(ref header, ref data))
            {
                Copy(ref header, ref data, ref headerNext, ref dataNext, header.count, math.min(depthCapacityRequested, header.depthCount));
                Dispose(ref header, ref data);
            }
            data = dataNext;
            header = headerNext;
        }

        public static void Add(ref KDTreeHeader header, ref KDTreeData data, int index, float3 position, float radius, Allocator allocator)
        {
            EnsureCapacity(ref header, ref data, header.count + 1, header.depthCapacity, allocator);

            data.payloadIndices[header.count] = index;
            data.positions[header.count] = position;
            data.radii[header.count] = radius;

            ++header.count;
        }

        [BurstCompile]
        public static bool TryAdd(ref KDTreeHeader header, ref KDTreeData data, int index, float3 position, float radius)
        {
            if (Capacity(ref header, ref data) < (header.count + 1)) { return false; }

            data.payloadIndices[header.count] = index;
            data.positions[header.count] = position;
            data.radii[header.count] = radius;

            ++header.count;

            return true;
        }

        [BurstCompile]
        public static bool IndexIsValid(ref KDTreeHeader header, ref KDTreeData data, int index)
        {
            return data.payloadIndices[index] != -1;
        }

        [BurstCompile]
        public static void RemoveAtPayloadIndex(ref KDTreeHeader header, ref KDTreeData data, int payloadIndex)
        {
            Debug.Assert(IsCreated(ref header, ref data));

            int index = FindIndexFromPayloadIndex(ref header, ref data, payloadIndex);
            Debug.Assert(index != -1);

            // Simply set our payload index to -1 to make KDTree treat this slot as a hole,
            // which will be skipped over when operating with the KDTree.
            // It is up to the user to query holeCount and decide when to rebuild the tree to improve query efficiency.
            ++header.holeCount;
            data.payloadIndices[index] = -1;
        }

        [BurstCompile]
        public static int FindIndexFromPayloadIndex(ref KDTreeHeader header, ref KDTreeData data, int payloadIndex)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(payloadIndex >= 0);

            for (int i = 0, iLen = header.count; i < iLen; ++i)
            {
                int payloadIndexCurrent = data.payloadIndices[i];
                if (payloadIndexCurrent == payloadIndex)
                {
                    return i;
                }
            }

            return -1;
        }

        [BurstCompile]
        public static void Clear(ref KDTreeHeader header, ref KDTreeData data)
        {
            header.count = 0;
            header.holeCount = 0;
        }

        [BurstCompile]
        public static bool IsLeaf(ref KDTreeHeader header, ref KDTreeData data, int depth)
        {
            Debug.Assert(IsCreated(ref header, ref data));

            return (depth + 1) == header.depthCount;
        }

        [BurstCompile]
        public static float ComputeSortKey(ref KDTreeHeader header, ref KDTreeData data, int i, int sortAxis)
        {
            Debug.Assert(i >= 0);
            Debug.Assert(i < header.count);
            Debug.Assert(sortAxis >= 0 && sortAxis <= 2);

            return IndexIsValid(ref header, ref data, i)
                ? data.positions[i][sortAxis]
                : float.MaxValue; // throw holes to the end of array.
        }

        [BurstCompile]
        public static void Swap(ref KDTreeHeader header, ref KDTreeData data, int indexA, int indexB)
        {
            Debug.Assert(indexA < Capacity(ref header, ref data));
            Debug.Assert(indexB < Capacity(ref header, ref data));

            (data.payloadIndices[indexA], data.payloadIndices[indexB]) = (data.payloadIndices[indexB], data.payloadIndices[indexA]);
            (data.positions[indexA], data.positions[indexB]) = (data.positions[indexB], data.positions[indexA]);
            (data.radii[indexA], data.radii[indexB]) = (data.radii[indexB], data.radii[indexA]);
        }

        [BurstCompile]
        private static void Quickselect(ref KDTreeHeader header, ref KDTreeData data, int left, int right, int n, int sortAxis)
        {
            //int debugLeft = left;
            //int debugRight = right;

            while (left != right)
            {

                // Partition:
                int indexC = ((right - left) >> 1) + left;
                float pivotData = ComputeSortKey(ref header, ref data, indexC, sortAxis);

                Swap(ref header, ref data, indexC, right);

                indexC = left;
                for (var j = left; j < right; ++j)
                {
                    if (ComputeSortKey(ref header, ref data, j, sortAxis) <= pivotData)
                    {

                        Swap(ref header, ref data, indexC, j);

                        ++indexC;
                    }
                }

                Swap(ref header, ref data, indexC, right);

                if (indexC == n)
                {
                    break;
                }

                if (n < indexC)
                {
                    right = indexC - 1;
                }
                else
                {
                    left = indexC + 1;
                }
                
                Debug.Assert(left >= 0);
                Debug.Assert(right >= 0);
                Debug.Assert(right >= left);
            }

            //// Debug:
            //float pivotKey = ComputeSortKey(ref data, n, sortAxis);
            //for (int d = debugLeft; d <= debugRight; ++d)
            //{
            //    if (d < n)
            //    {
            //        if (!(ComputeSortKey(ref data, d, sortAxis) <= pivotKey))
            //        {
            //            Debug.Assert(false);
            //        }
            //    }
            //    else if (d > n)
            //    {
            //        if (!(ComputeSortKey(ref data, d, sortAxis) >= pivotKey))
            //        {
            //            Debug.Assert(false);
            //        }
            //    }
            //}
        }

        [BurstCompile]
        public static void ComputeAABBFromRange(out float3 aabbMin, out float3 aabbMax, ref KDTreeHeader header, ref KDTreeData data, int left, int right)
        {
            aabbMin = float.MaxValue;
            aabbMax = -float.MaxValue;

            Debug.Assert(left >= 0 && left < header.count);
            Debug.Assert(right >= 0 && right < header.count);

            for (int i = left; i <= right; ++i)
            {
                if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
                if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
                if (float.IsNaN(data.radii[i])) { continue; }

                aabbMin = math.min(aabbMin, data.positions[i] - data.radii[i]);
                aabbMax = math.max(aabbMax, data.positions[i] + data.radii[i]);
            }
        }

        [BurstCompile]
        private static int ComputeSplitAxisFromAABB(float3 aabbMin, float3 aabbMax)
        {
            float3 extents = aabbMax - aabbMin;

            return (extents[0] > extents[1])
                ? ((extents[0] > extents[2]) ? 0 : 2)
                : ((extents[1] > extents[2]) ? 1 : 2);
        }

        [BurstCompile]
        public static void Build(ref KDTreeHeader header, ref KDTreeData data, int depthCount)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCapacity >= depthCount);
            Debug.Assert(depthCount > 0);

            CompactHoles(ref header, ref data);

            // data.count can reach zero after CompactHoles is executed.
            if (header.count == 0) { return; }

            header.depthCount = math.clamp(depthCount, 1, CeilLog2(header.count));

            int nodeIndex = 0;
            for (int d = 0, dLen = depthCount; d < dLen; ++d)
            {
                int nodeCountAtDepth = 1 << d;
                for (int n = 0; n < nodeCountAtDepth; ++n)
                {
                    // x >> d == x / (1 << d)
                    uint numBitsNeededHeaderCount = (header.count > 0) ? CeilLog2((uint)header.count) : 0;
                    uint numBitsNeededNode = (n > 0) ? CeilLog2((uint)n) : 0;
                    uint numBitsNeeded = numBitsNeededHeaderCount + numBitsNeededNode;
                    Debug.Assert(numBitsNeeded < 32); // left and right calculations can underflow with large header.counts and node counts.
                    
                    int left = ((n + 0) * header.count) >> d;
                    int right = (((n + 1) * header.count) >> d) - 1;

                    Debug.Assert(left >= 0 && left < header.count);
                    Debug.Assert(right >= 0 && right < header.count);
                    Debug.Assert(right >= left);

                    ComputeAABBFromRange(out float3 aabbMin, out float3 aabbMax, ref header, ref data, left, right);

                    data.aabbs[nodeIndex * 2 + 0] = aabbMin;
                    data.aabbs[nodeIndex * 2 + 1] = aabbMax;

                    // Only perform split if we are not leaf node.
                    if ((d + 1) < depthCount)
                    {
                        int sortAxis = ComputeSplitAxisFromAABB(aabbMin, aabbMax);

                        int pivot = ((right - left) >> 1) + left;
                        Debug.Assert(left <= pivot);
                        Debug.Assert(pivot <= right);

                        Quickselect(ref header, ref data, left, right, pivot, sortAxis);

                        data.splitAxes[nodeIndex] = (byte)sortAxis;
                        data.splitPositions[nodeIndex] = data.positions[pivot][sortAxis];
                    }
                    

                    ++nodeIndex;
                }
            }        
        }

        [BurstCompile]
        public static void CompactHoles(ref KDTreeHeader header, ref KDTreeData data)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);

            if (header.holeCount == 0) { return; }

            int left = 0;
            int right = header.count - 1;
            int pivot = right - header.holeCount;
            int sortAxis = 0; // Sort axis can be any axis, we only care about partitioning holes to the end.
            Quickselect(ref header, ref data, left, right, pivot, sortAxis);

            // Debug:
            for (int i = pivot + 1; i <= right; ++i)
            {
                Debug.Assert(!IndexIsValid(ref header, ref data, i));
            }

            // All holes are now partitioned to the end of the array.
            // We can shink counts and clear out holes.
            header.count -= header.holeCount;
            header.holeCount = 0;
        }

        [BurstCompile]
        public static void FindNearestNeighborStackless(out int index, out float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, float distanceSquaredMax = float.MaxValue)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            index = -1;
            distanceSquared = distanceSquaredMax;

            KDTreeTraversalState state = KDTreeTraversalState.zero;
            while (FindNearestNeighborStacklessIterator(out int left, out int right, ref distanceSquared, ref state, ref header, ref data, position))
            {
                if (left == -1) { continue; }
                
                Debug.Assert(left >= 0 && left < header.count);
                Debug.Assert(right >= 0 && right < header.count);

                for (int i = left; i <= right; ++i)
                {
                    if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
                    if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
                    if (float.IsNaN(data.radii[i])) { continue; }

                    float distanceCurrent = math.max(0.0f, math.length(data.positions[i] - position) - data.radii[i]);
                    float distanceSquaredCurrent = distanceCurrent * distanceCurrent;
                    if (distanceSquaredCurrent < distanceSquared)
                    {
                        distanceSquared = distanceSquaredCurrent;
                        index = i;
                    }
                }
            }
        }

        [BurstCompile]
        public static void FindNearestNeighborStacklessGeneralizedLeftFirst(out int index, out float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, float distanceSquaredMax = float.MaxValue)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            index = -1;
            distanceSquared = distanceSquaredMax;

            KDTreeTraversalStateGeneralized state = KDTreeTraversalStateGeneralized.zero;
            while (FindNearestNeighborStacklessGeneralizedLeftFirstIterator(out int left, out int right, distanceSquared, ref state, ref header, ref data, position))
            {
                Debug.Assert(left >= 0 && left < header.count);
                Debug.Assert(right >= 0 && right < header.count);

                for (int i = left; i <= right; ++i)
                {
                    if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
                    if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
                    if (float.IsNaN(data.radii[i])) { continue; }

                    float distanceCurrent = math.max(0.0f, math.length(data.positions[i] - position) - data.radii[i]);
                    float distanceSquaredCurrent = distanceCurrent * distanceCurrent;
                    if (distanceSquaredCurrent < distanceSquared)
                    {
                        distanceSquared = distanceSquaredCurrent;
                        index = i;
                    }
                }
            }
        }

        // TODO: There seem to be bugs with the StacklessGeneralizedIterator - so we need to use FindNearestNeighborStacklessGeneralizedLeftFirstIterator for now (which is less optimal).
        // [BurstCompile]
        // public static void FindNearestNeighborStacklessGeneralized(out int index, out float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, float distanceSquaredMax = float.MaxValue)
        // {
        //     Debug.Assert(IsCreated(ref header, ref data));
        //     Debug.Assert(header.count > 0);
        //     Debug.Assert(header.depthCount > 0);

        //     index = -1;
        //     distanceSquared = distanceSquaredMax;

        //     KDTreeTraversalStateGeneralized state = KDTreeTraversalStateGeneralized.zero;
        //     while (FindNearestNeighborStacklessGeneralizedIterator(out int left, out int right, ref distanceSquared, ref state, ref header, ref data, position))
        //     {
        //         Debug.Assert(left >= 0 && left < header.count);
        //         Debug.Assert(right >= 0 && right < header.count);

        //         for (int i = left; i <= right; ++i)
        //         {
        //             if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
        //             if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
        //             if (float.IsNaN(data.radii[i])) { continue; }

        //             float distanceCurrent = math.max(0.0f, math.length(data.positions[i] - position) - data.radii[i]);
        //             float distanceSquaredCurrent = distanceCurrent * distanceCurrent;
        //             if (distanceSquaredCurrent < distanceSquared)
        //             {
        //                 distanceSquared = distanceSquaredCurrent;
        //                 index = i;
        //             }
        //         }
        //     }
        // }

        [BurstCompile]
        public static void FindNearestNeighbor(out int index, out float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, float distanceSquaredMax = float.MaxValue)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            index = -1;
            distanceSquared = distanceSquaredMax;
            FindNearestNeighborRecursive(ref index, ref distanceSquared, ref header, ref data, position, 0, 0);
        }

        [BurstCompile]
        private static void FindNearestNeighborRecursive(ref int index, ref float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, int depth, int nodeIndexAtDepth)
        {
            if (IsLeaf(ref header, ref data, depth))
            {
                FindNearestNeighborInLeaf(ref index, ref distanceSquared, ref header, ref data, position, depth, nodeIndexAtDepth);
            }
            else
            {
                FindNearestNeighborInBranches(ref index, ref distanceSquared, ref header, ref data, position, depth, nodeIndexAtDepth);
            }
        }

        [BurstCompile]
        private static void FindNearestNeighborInLeaf(ref int index, ref float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, int depth, int nodeIndexAtDepth)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            Debug.Assert(IsLeaf(ref header, ref data, depth));

            // Leaf node.
            // Loop over all contained elements and find nearest point.
            ComputeNodeIndexRange(out int left, out int right, ref header, ref data, depth, nodeIndexAtDepth);

            Debug.Assert(left >= 0 && left < header.count);
            Debug.Assert(right >= 0 && right < header.count);

            for (int i = left; i <= right; ++i)
            {
                if (!IndexIsValid(ref header, ref data, i)) { continue; } // Item has been removed. Skip.
                if (float.IsNaN(data.positions[i].x) || float.IsNaN(data.positions[i].y) || float.IsNaN(data.positions[i].z)) { continue; }
                if (float.IsNaN(data.radii[i])) { continue; }

                float distanceCurrent = math.max(0.0f, math.length(data.positions[i] - position) - data.radii[i]);
                float distanceSquaredCurrent = distanceCurrent * distanceCurrent;
                if (distanceSquaredCurrent < distanceSquared)
                {
                    distanceSquared = distanceSquaredCurrent;
                    index = i;
                }
            }
        }

        [BurstCompile]
        private static void FindNearestNeighborInBranches(ref int index, ref float distanceSquared, ref KDTreeHeader header, ref KDTreeData data, float3 position, int depth, int nodeIndexAtDepth)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            Debug.Assert(!IsLeaf(ref header, ref data, depth));

            // Node has children.
            // Recurse into nearest child.
            // (1 << 0) == 1
            // (1 << 1) == 2
            // (1 << 2) == 4
            // (1 << 4) == 8
            //
            //          0
            //    1            2
            //  3   4       5     6
            // 7 8 9 10   11 12 13 14
            int nodeIndexParent = ComputeNodeParent(ref header, ref data, depth, nodeIndexAtDepth);

            int sortAxisParent = data.splitAxes[nodeIndexParent];
            float splitPlaneParent = data.splitPositions[nodeIndexParent];
            float splitPlaneDistanceSigned = position[sortAxisParent] - splitPlaneParent;

            int childDepth = depth + 1;
            int childLeftNodeIndexAtDepth = nodeIndexAtDepth * 2 + 0;
            int childRightNodeIndexAtDepth = nodeIndexAtDepth * 2 + 1;

            // Use split plane as heristic for deciding which child to enter first.
            int childNearNodeIndexAtDepth = (splitPlaneDistanceSigned <= 0.0f) ? childLeftNodeIndexAtDepth : childRightNodeIndexAtDepth;
            int childFarNodeIndexAtDepth = (splitPlaneDistanceSigned <= 0.0f) ? childRightNodeIndexAtDepth : childLeftNodeIndexAtDepth;

            // Use actual AABBs to determine whether or not we need to enter the child.
            {

                int nodeIndexChild = ComputeNodeParent(ref header, ref data, childDepth, childNearNodeIndexAtDepth);
                float3 aabbMin = data.aabbs[nodeIndexChild * 2 + 0];
                float3 aabbMax = data.aabbs[nodeIndexChild * 2 + 1];
                float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                float3 sampleMin = position - new float3(distance, distance, distance);
                float3 sampleMax = position + new float3(distance, distance, distance);

                if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                {
                    FindNearestNeighborRecursive(ref index, ref distanceSquared, ref header, ref data, position, childDepth, childNearNodeIndexAtDepth);
                }
            }

            {

                int nodeIndexChild = ComputeNodeParent(ref header, ref data, childDepth, childFarNodeIndexAtDepth);
                float3 aabbMin = data.aabbs[nodeIndexChild * 2 + 0];
                float3 aabbMax = data.aabbs[nodeIndexChild * 2 + 1];
                float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                float3 sampleMin = position - new float3(distance, distance, distance);
                float3 sampleMax = position + new float3(distance, distance, distance);

                if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                {
                    FindNearestNeighborRecursive(ref index, ref distanceSquared, ref header, ref data, position, childDepth, childFarNodeIndexAtDepth);
                }
            }
        }

        [BurstCompile]
        public static int ComputeNodeParent(ref KDTreeHeader header, ref KDTreeData data, int depth, int nodeIndexAtDepth)
        {
            return (1 << depth) - 1 + nodeIndexAtDepth;
        }

        [BurstCompile]
        public static void ComputeNodeIndexRange(out int left, out int right, ref KDTreeHeader header, ref KDTreeData data, int depth, int nodeIndexAtDepth)
        {
            // x >> d == x / (1 << depth)
            left = ((nodeIndexAtDepth + 0) * header.count) >> depth;
            right = (((nodeIndexAtDepth + 1) * header.count) >> depth) - 1;
        }

        [BurstCompile]
        public struct KDTreeTraversalState
        {
            public enum Direction : byte
            {
                Down = 0,
                Up,
                Side
            };

            public Direction direction;
            public int depth;
            public int nodeIndexAtDepth;

            public static readonly KDTreeTraversalState zero = new KDTreeTraversalState
            {
                direction = Direction.Down,
                depth = 0,
                nodeIndexAtDepth = 0
            };
        }

        [BurstCompile]
        public static bool FindNearestNeighborStacklessIterator(out int left, out int right, ref float distanceSquared, ref KDTreeTraversalState state, ref KDTreeHeader header, ref KDTreeData data, float3 position)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);
            Debug.Assert(header.depthCount > state.depth);
            Debug.Assert(state.depth >= 0);

            left = -1;
            right = -1;

            KDTreeTraversalState stateNextUp = new KDTreeTraversalState()
            {
                direction = KDTreeTraversalState.Direction.Up,
                depth = state.depth - 1,
                nodeIndexAtDepth = state.nodeIndexAtDepth >> 1
            };

            KDTreeTraversalState stateNextDownLeft = new KDTreeTraversalState()
            {
                direction = KDTreeTraversalState.Direction.Down,
                depth = state.depth + 1,
                nodeIndexAtDepth = (state.nodeIndexAtDepth << 1) + 0
            };

            KDTreeTraversalState stateNextDownRight = new KDTreeTraversalState()
            {
                direction = KDTreeTraversalState.Direction.Down,
                depth = state.depth + 1,
                nodeIndexAtDepth = (state.nodeIndexAtDepth << 1) + 1
            };

            KDTreeTraversalState stateNextRight = new KDTreeTraversalState()
            {
                direction = KDTreeTraversalState.Direction.Side,
                depth = state.depth,
                nodeIndexAtDepth = state.nodeIndexAtDepth + 1
            };

            KDTreeTraversalState stateNextLeft = new KDTreeTraversalState()
            {
                direction = KDTreeTraversalState.Direction.Side,
                depth = state.depth,
                nodeIndexAtDepth = state.nodeIndexAtDepth - 1
            };

            if (state.direction == KDTreeTraversalState.Direction.Up)
            {
                if (state.depth == 0)
                {
                    // Finished traversing back up to root. Done.
                    return false;
                }


                if (IsNodeLeft(state.nodeIndexAtDepth))
                {
                    // At left node. Move right.
                    state = stateNextRight;
                    return true;
                }
                else
                {
                    // At right node. Move up.
                    state = stateNextUp;
                    return true;
                }
            }
            else // Down or Side
            {
                if (IsLeaf(ref header, ref data, state.depth))
                {
                    // Leaf node.
                    // Return range to loop over all contained elements and find nearest point.
                    // Perform an AABB check here to early out.
                    // Previous aabb checks during the traversal simply test which side of the split plane we are on.
                    // TODO: Rather than AABB against AABB intersection, a more accurate cull would be AABB against sphere.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    int nodeIndexParent = ComputeNodeParent(ref header, ref data, state.depth, state.nodeIndexAtDepth);
                    float3 aabbMin = data.aabbs[nodeIndexParent * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndexParent * 2 + 1];
                    float3 sampleMin = position - new float3(distance, distance, distance);
                    float3 sampleMax = position + new float3(distance, distance, distance);
                    
                    if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                    {
                        ComputeNodeIndexRange(out left, out right, ref header, ref data, state.depth, state.nodeIndexAtDepth);
                    }

                    if (state.direction == KDTreeTraversalState.Direction.Side)
                    {
                        // Came from the side. Time to move up.
                        state = stateNextUp;
                        return true;
                    }
                    else
                    {
                        // Came from above. Time to move to the side.
                        Debug.Assert(state.direction == KDTreeTraversalState.Direction.Down);
                        // state = IsNodeLeft(state.nodeIndexAtDepth) ? stateNextRight : stateNextLeft;
                        //Debug.Assert(IsNodeLeft(state.nodeIndexAtDepth)); // Always traverse into left first.
                        state = stateNextRight;
                        return true;
                    }
                }
                else
                {
                    // Node has children.
                    // Recurse into nearest child.
                    // (1 << 0) == 1
                    // (1 << 1) == 2
                    // (1 << 2) == 4
                    // (1 << 4) == 8
                    //
                    //          0
                    //    1            2
                    //  3   4       5     6
                    // 7 8 9 10   11 12 13 14
                    int nodeIndexParent = ComputeNodeParent(ref header, ref data, state.depth, state.nodeIndexAtDepth);
                    int sortAxisParent = data.splitAxes[nodeIndexParent];
                    float splitPlaneParent = data.splitPositions[nodeIndexParent];
                    float splitPlaneDistanceSigned = position[sortAxisParent] - splitPlaneParent;
                    // float splitPlaneDistanceSquared = splitPlaneDistanceSigned * splitPlaneDistanceSigned;

                    // KDTreeTraversalState stateNextNear = (splitPlaneDistanceSigned <= 0.0f) ? stateNextDownLeft : stateNextDownRight;
                    // KDTreeTraversalState stateNextFar = (splitPlaneDistanceSigned <= 0.0f) ? stateNextDownRight : stateNextDownLeft;

                    // Always traverse down left first.
                    state = stateNextDownLeft;
                    return true;

                    //if (state.direction == KDTreeTraversalState.Direction.Side)
                    //{
                    //    // Came from the side. We are in the right node.
                    //    // Move down left.
                    //    state = stateNextDownLeft;
                    //    //
                    //    // Potential early out: if distance to the split plane is greater than our current nearest distance,
                    //    // we know that no points within opposite split plane can be closer.
                    //    //if (splitPlaneDistanceSigned >= 0.0f || splitPlaneDistanceSquared < distanceSquared)
                    //    //{
                    //    //    state = stateNextDownRight;
                    //    //    return true;
                    //    //}
                    //    //else
                    //    //{
                    //    //    state = stateNextUp;
                    //    //    return true;
                    //    //}
                    //}
                    //else
                    //{
                    //    // Came from above. We are in the left node.
                    //    // Move down right.
                    //    Debug.Assert(state.direction == KDTreeTraversalState.Direction.Down);

                    //    state = stateNextDownRight;

                    //    //if (splitPlaneDistanceSigned < 0.0f || splitPlaneDistanceSquared < distanceSquared)
                    //    //{
                    //    //    state = stateNextDownLeft;
                    //    //    return true;
                    //    //}
                    //    //else
                    //    //{
                    //    //    // Skip left side, and go straight to right side.
                    //    //    // Need to encode it as if we came from the left for our state machine to work.
                    //    //    state = stateNextDownRight;
                    //    //    state.direction = KDTreeTraversalState.Direction.Side;
                    //    //    return true;
                    //    //}
                    //}
                }
            }

            // Should never reach here.
            //Debug.Assert(false);
        }

        // http://jcgt.org/published/0002/01/03/paper.pdf
        [BurstCompile]
        public struct KDTreeTraversalStateGeneralized
        {
            public uint levelStart;
            public uint levelIndex;
            public uint swapMask;

            public static readonly KDTreeTraversalStateGeneralized zero = new KDTreeTraversalStateGeneralized
            {
                levelStart = 1,
                levelIndex = 0,
                swapMask = 0
            };
        }

        [BurstCompile]
        public static bool FindNearestNeighborStacklessGeneralizedLeftFirstIterator(out int left, out int right, float distanceSquared, ref KDTreeTraversalStateGeneralized state, ref KDTreeHeader header, ref KDTreeData data, float3 position)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);
            // Debug.Assert(header.depthCount > state.depth);
            // Debug.Assert(state.depth >= 0);

            left = -1;
            right = -1;

            while (state.levelStart >= 1)
            {
                int nodeIndex = (int)(state.levelStart + state.levelIndex - 1u);

                int depth = (int)CeilLog2(state.levelStart); // TODO: Fixme!

                if (IsLeaf(ref header, ref data, (int)depth))
                {
                    // Leaf node.
                    // Return range to loop over all contained elements and find nearest point.
                    // Perform an AABB check here to early out.
                    // Previous aabb checks during the traversal simply test which side of the split plane we are on.
                    // TODO: Rather than AABB against AABB intersection, a more accurate cull would be AABB against sphere.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 aabbMin = data.aabbs[nodeIndex * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndex * 2 + 1];
                    float3 sampleMin = position - new float3(distance, distance, distance);
                    float3 sampleMax = position + new float3(distance, distance, distance);

                    if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                    {
                        ComputeNodeIndexRange(out left, out right, ref header, ref data, (int)depth, (int)state.levelIndex);

                        // Move up a level.
                        state.levelIndex = state.levelIndex + 1;
                        uint up = CountTrailingZeros(state.levelIndex);
                        state.levelStart >>= (int)up;
                        state.levelIndex >>= (int)up;
                        state.levelStart = (state.levelStart == 1) ? 0 : state.levelStart; // Signal if we are done, back at root.
                        return true; // More to do.
                    }
                }
                else
                {
                    // Branch
                    // Node has children.
                    // Recurse into nearest child.
                    // (1 << 0) == 1
                    // (1 << 1) == 2
                    // (1 << 2) == 4
                    // (1 << 4) == 8
                    //
                    //          0
                    //    1            2
                    //  3   4       5     6
                    // 7 8 9 10   11 12 13 14
                    float3 aabbMin = data.aabbs[nodeIndex * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndex * 2 + 1];

                    // Use actual AABBs to determine whether or not we need to enter the child.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 sampleMin = position - new float3(distance, distance, distance);
                    float3 sampleMax = position + new float3(distance, distance, distance);

                    if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                    {
                        state.levelStart <<= 1;
                        state.levelIndex <<= 1;
                        continue;
                    }
                }

                {
                    state.levelIndex = state.levelIndex + 1;
                    uint up = CountTrailingZeros(state.levelIndex);
                    state.levelStart >>= (int)up;
                    state.levelIndex >>= (int)up;
                    if (state.levelStart == 1) { return false; }
                }
            }

            return false;
        }
        
        // http://jcgt.org/published/0002/01/03/paper.pdf
        [BurstCompile]
        public struct KDTreeTraversalStateGeneralizedOptimized
        {
            public uint node;
            public uint traversal;

            public static readonly KDTreeTraversalStateGeneralizedOptimized zero = new KDTreeTraversalStateGeneralizedOptimized
            {
                node = 1,
                traversal = 0
            };
        }

        [BurstCompile]
        public static bool FindNearestNeighborStacklessGeneralizedIteratorOptimized(out int left, out int right, float distanceSquared, ref KDTreeTraversalStateGeneralizedOptimized state, ref KDTreeHeader header,
            ref KDTreeData data, float3 position)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            left = -1;
            right = -1;
            
            while (state.node > 0)
            {
                int nodeIndex = (int)(state.node - 1u);

                int depth = (state.node == 0) ? 0 : (int)FloorLog2((uint)state.node);

                if (IsLeaf(ref header, ref data, (int)depth))
                {
                    // Leaf node.
                    // Return range to loop over all contained elements and find nearest point.
                    // Perform an AABB check here to early out.
                    // Previous aabb checks during the traversal simply test which side of the split plane we are on.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 aabbMin = data.aabbs[nodeIndex * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndex * 2 + 1];
                    if (ComputeAABBContainsSphere(aabbMin, aabbMax, position, distance))
                    {
                        {
                            left = ((nodeIndex - (1 << depth) + 1 + 0) * header.count) >> depth;
                            right = (((nodeIndex - (1 << depth) + 1 + 1) * header.count) >> depth) - 1;
                        }

                        // Leaf node found, inline traversal state update before returning
                        ++state.traversal;
                        int up = (int)CountTrailingZeros(state.traversal);
                        state.traversal >>= up;
                        state.node >>= up;
                        return true;
                    }
                }
                else
                {
                    bool isAnyChildrenAccepted = false;
                    bool isRightChildFirst = false;
                    bool isRejectOneChild = false;
                    
                    // Branch
                    // Node has children.
                    // Recurse into nearest child.
                    // (1 << 0) == 1
                    // (1 << 1) == 2
                    // (1 << 2) == 4
                    // (1 << 4) == 8
                    //
                    //          0
                    //    1            2
                    //  3   4       5     6
                    // 7 8 9 10   11 12 13 14
                    int sortAxisParent = data.splitAxes[nodeIndex];
                    float splitPlaneParent = data.splitPositions[nodeIndex];
                    float splitPlaneDistanceSigned = position[sortAxisParent] - splitPlaneParent;

                    int childNodeIndexAtDepthLeft = (int)(state.node << 1) - 1;
                    int childNodeIndexAtDepthRight = childNodeIndexAtDepthLeft + 1;

                    // Use split plane as heuristic for deciding which child to enter first.
                    isRightChildFirst = splitPlaneDistanceSigned > 0.0f;

                    // Use actual AABBs to determine whether or not we need to enter the child.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 sampleMin = position - new float3(distance, distance, distance);
                    float3 sampleMax = position + new float3(distance, distance, distance);
                    {

                        int nodeIndexChild = childNodeIndexAtDepthLeft;
                        float3 aabbMin = data.aabbs[nodeIndexChild * 2 + 0];
                        float3 aabbMax = data.aabbs[nodeIndexChild * 2 + 1];

                        if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                        {
                            isAnyChildrenAccepted = true;
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }

                    {

                        int nodeIndexChild = childNodeIndexAtDepthRight;
                        float3 aabbMin = data.aabbs[nodeIndexChild * 2 + 0];
                        float3 aabbMax = data.aabbs[nodeIndexChild * 2 + 1];

                        if (ComputeAABBContainsAABB(aabbMin, aabbMax, sampleMin, sampleMax))
                        {
                            isAnyChildrenAccepted = true;
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }
			
                    if (isAnyChildrenAccepted)
                    {
                        state.node += state.node + (isRightChildFirst ? 1u : 0u);
                        state.traversal += state.traversal + (isRejectOneChild ? 1u : 0u);
                        continue;
                    }
                }

                bool isDone = false;
                {
                    ++state.traversal;
                    int up = (int)CountTrailingZeros(state.traversal);

                    state.traversal >>= up;
                    state.node >>= up;
		
                    isDone = state.node <= 1;

                    state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                }


                if (isDone)
                {
                    return false;
                }
            }

            return false;
        }
        
        [BurstCompile]
        public static bool FindRayIntersectionStacklessGeneralizedIteratorOptimized(out int left, out int right, float distanceSquared, ref KDTreeTraversalStateGeneralizedOptimized state, ref KDTreeHeader header,
            ref KDTreeData data, float3 rayOrigin, float3 rayDirectionInverse)
        {
            Debug.Assert(IsCreated(ref header, ref data));
            Debug.Assert(header.count > 0);
            Debug.Assert(header.depthCount > 0);

            left = -1;
            right = -1;
            
            while (state.node > 0)
            {
                int nodeIndex = (int)(state.node - 1u);

                int depth = (state.node == 0) ? 0 : (int)FloorLog2((uint)state.node);

                if (IsLeaf(ref header, ref data, (int)depth))
                {
                    // Leaf node.
                    // Return range to loop over all contained elements and find nearest point.
                    // Perform an AABB check here to early out.
                    // Previous aabb checks during the traversal simply test which side of the split plane we are on.
                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float3 aabbMin = data.aabbs[nodeIndex * 2 + 0];
                    float3 aabbMax = data.aabbs[nodeIndex * 2 + 1];
                    if (ComputeAABBContainsSphere(aabbMin, aabbMax, rayOrigin, distance))
                    {
                        {
                            left = ((nodeIndex - (1 << depth) + 1 + 0) * header.count) >> depth;
                            right = (((nodeIndex - (1 << depth) + 1 + 1) * header.count) >> depth) - 1;
                        }

                        // Leaf node found, inline traversal state update before returning
                        ++state.traversal;
                        int up = (int)CountTrailingZeros(state.traversal);
                        state.traversal >>= up;
                        state.node >>= up;
                        state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                        return true;
                    }
                }
                else
                {
                    bool isAnyChildrenAccepted = false;
                    bool isRightChildFirst = false;
                    bool isRejectOneChild = false;
                    
                    // Branch
                    // Node has children.
                    // Recurse into nearest child.
                    // (1 << 0) == 1
                    // (1 << 1) == 2
                    // (1 << 2) == 4
                    // (1 << 4) == 8
                    //
                    //          0
                    //    1            2
                    //  3   4       5     6
                    // 7 8 9 10   11 12 13 14
                    int childNodeIndexLeft = (int)(state.node << 1) - 1;
                    int childNodeIndexRight = childNodeIndexLeft + 1;

                    float distance = (distanceSquared > 1e-5f) ? math.sqrt(distanceSquared) : 0.0f;
                    float childLeftT0 = 0.0f;
                    float childLeftT1 = distance;
                    {
                        float3 aabbMin = data.aabbs[childNodeIndexLeft * 2 + 0];
                        float3 aabbMax = data.aabbs[childNodeIndexLeft * 2 + 1];

                        if (IntersectRayAABB(ref childLeftT0, ref childLeftT1, rayOrigin, rayDirectionInverse, aabbMin, aabbMax))
                        {
                            isAnyChildrenAccepted = true;
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }
                    
                    float childRightT0 = 0.0f;
                    float childRightT1 = distance;
                    {
                        float3 aabbMin = data.aabbs[childNodeIndexRight * 2 + 0];
                        float3 aabbMax = data.aabbs[childNodeIndexRight * 2 + 1];
                        
                        if (IntersectRayAABB(ref childRightT0, ref childRightT1, rayOrigin, rayDirectionInverse, aabbMin, aabbMax))
                        {
                            isAnyChildrenAccepted = true;
                            isRightChildFirst = isRejectOneChild
                                ? true
                                : ((childLeftT0 == childRightT0)
                                    ? (childRightT1 < childLeftT1)
                                    : (childRightT0 < childLeftT0));
                        }
                        else
                        {
                            isRejectOneChild = true;
                        }
                    }

                    if (isAnyChildrenAccepted)
                    {
                        state.node += state.node + (isRightChildFirst ? 1u : 0u);
                        state.traversal += state.traversal + (isRejectOneChild ? 1u : 0u);
                        continue;
                    }
                }

                bool isDone = false;
                {
                    ++state.traversal;
                    int up = (int)CountTrailingZeros(state.traversal);

                    state.traversal >>= up;
                    state.node >>= up;
		
                    isDone = state.node <= 1;

                    state.node = state.node + 1 - ((state.node & 1) << 1); // sibling
                }


                if (isDone)
                {
                    return false;
                }
            }

            return false;
        }
        
        // https://jcgt.org/published/0002/02/02/
        [BurstCompile]
        private static bool IntersectRayAABB( ref float tmin, ref float tmax, float3 rayOrigin, float3 rayDirectionInverse, float3 aabbMin, float3 aabbMax)
        {
            float txmin, txmax, tymin, tymax, tzmin, tzmax;
            txmin = (((rayDirectionInverse.x < 0) ? aabbMax.x : aabbMin.x) - rayOrigin.x) * rayDirectionInverse.x;
            txmax = (((rayDirectionInverse.x < 0) ? aabbMin.x : aabbMax.x) - rayOrigin.x) * rayDirectionInverse.x;
            tymin = (((rayDirectionInverse.y < 0) ? aabbMax.y : aabbMin.y) - rayOrigin.y) * rayDirectionInverse.y;
            tymax = (((rayDirectionInverse.y < 0) ? aabbMin.y : aabbMax.y) - rayOrigin.y) * rayDirectionInverse.y;
            tzmin = (((rayDirectionInverse.z < 0) ? aabbMax.z : aabbMin.z) - rayOrigin.z) * rayDirectionInverse.z;
            tzmax = (((rayDirectionInverse.z < 0) ? aabbMin.z : aabbMax.z) - rayOrigin.z) * rayDirectionInverse.z;
            
            tmin = math.max(tzmin, math.max(tymin, math.max(txmin, tmin)));
            tmax = math.min(tzmax, math.min(tymax, math.min(txmax, tmax)));
            tmax *= 1.00000024f;
            return tmin <= tmax;
        }
        
        [BurstCompile]
        // Branchless implementation
        public static bool IntersectRayTriangle(
            out float t,
            out float triangleBarycentricA,
            out float triangleBarycentricB,
            out float triangleBarycentricC,
            float3 rayOrigin, float3 rayDirection, float tmin, float tmax,
            float3 triangleVertexPositionA, float3 triangleVertexPositionB, float3 triangleVertexPositionC
        )
        {
            float3 e0 = triangleVertexPositionB - triangleVertexPositionA;
            float3 e1 = triangleVertexPositionA - triangleVertexPositionC;
            float3 n  = math.cross(e1, e0);

            float3 e2 = (1.0f / math.dot( n, rayDirection)) * (triangleVertexPositionA - rayOrigin);
            float3 i  = math.cross(rayDirection, e2 );

            triangleBarycentricB  = math.dot( i, e1 );
            triangleBarycentricC = math.dot( i, e0 );
            triangleBarycentricA = 1.0f - triangleBarycentricB - triangleBarycentricC;
            t = math.dot( n, e2 );

            return
                (t < tmax)
                && (t > tmin) 
                && (triangleBarycentricB >= 0.0f)
                && (triangleBarycentricC >= 0.0f)
                && ((triangleBarycentricB + triangleBarycentricC) <= 1.0f);
        }

        [BurstCompile]
        private static bool IsNodeLeft(int nodeIndexAtDepth)
        {
            return (((uint)nodeIndexAtDepth) & 1u) == 0;
        }

        [BurstCompile]
        private static bool ComputeAABBContainsPosition(float3 aabbMin, float3 aabbMax, float3 position)
        {
            return !(math.any(position < aabbMin) || math.any(position > aabbMax));
        }

        [BurstCompile]
        private static bool ComputeAABBContainsAABB(float3 aabbLHSMin, float3 aabbLHSMax, float3 aabbRHSMin, float3 aabbRHSMax)
        {
            return !(math.any(aabbRHSMax < aabbLHSMin) || math.any(aabbRHSMin > aabbLHSMax));
        }
        
        [BurstCompile]
        private static bool ComputeAABBContainsSphere(float3 aabbMin, float3 aabbMax, float3 spherePosition, float sphereRadius)
        {
            float distanceSquared = 0.0f;
            for( int i = 0; i < 3; i++ )
            {
                float v = spherePosition[i];
                if( v < aabbMin[i] ) distanceSquared += (aabbMin[i] - v) * (aabbMin[i] - v);
                if( v > aabbMax[i] ) distanceSquared += (v - aabbMax[i]) * (v - aabbMax[i]);
            }

            return distanceSquared <= (sphereRadius * sphereRadius);
        }

        [BurstCompile]
        public static float4 ComputeDebugColorFromNodeIndex(int nodeIndex, int nodeCount)
        {
            // Spiral through chroma space.
            // Constant luminance.

            float radiusNormalized = ((float)nodeIndex + 0.5f) / (float)nodeCount;
            float theta = radiusNormalized * 2.0f * math.PI;
            float radius = radiusNormalized; // chroma space is in [0.0f, 1.0] range.
            float2 chroma = new float2(
                math.cos(theta),
                math.sin(theta)
            ) * radius;

            chroma += 0.5f;

            float3 ycocg = new float3(1.0f, chroma);
            float3 rgb;
            rgb.y = ycocg.x + ycocg.z;
            float temp = ycocg.x - ycocg.z;
            rgb.x = temp + ycocg.y;
            rgb.z = temp - ycocg.y;

            return new float4(rgb, 1.0f);
        }

        [BurstCompile]
        public static int ComputeDepthCountFromElements(int elementCount, int elementsPerLeaf)
        {
            return math.max(1, CeilLog2((elementCount + elementsPerLeaf - 1) / elementsPerLeaf));
        }

        // Basic implementation of ceil(log2(x)) which does not handle case of x == 0.
        [BurstCompile]
        private static uint CeilLog2(uint x)
        {
            Debug.Assert(x > 0);
            uint i = 0;
            uint y = x;
            if (y >= (1u << 16)) { i += 16u; y >>= 16; }
            if (y >= (1u << 8)) { i += 8u; y >>= 8; }
            if (y >= (1u << 4)) { i += 4u; y >>= 4; }
            if (y >= (1u << 2)) { i += 2u; y >>= 2; }
            if (y >= (1u << 1)) { i += 1u; y >>= 1; }
            return (x > (1u << (int)i)) ? (i + 1u) : i;
        }
        
        [BurstCompile]
        private static uint FloorLog2(uint x)
        {
            Debug.Assert(x > 0);
            uint i = 0;
            uint y = x;
    
            if (y >= (1u << 16)) { i += 16u; y >>= 16; }
            if (y >= (1u << 8)) { i += 8u; y >>= 8; }
            if (y >= (1u << 4)) { i += 4u; y >>= 4; }
            if (y >= (1u << 2)) { i += 2u; y >>= 2; }
            if (y >= (1u << 1)) { i += 1u; }

            return i;
        }

        [BurstCompile]
        private static int CeilLog2(int x)
        {
            Debug.Assert(x >= 0);

            return (int)CeilLog2((uint)x);
        }

        // Hackers Delight Figure 5-14
        [BurstCompile]
        private static uint CountTrailingZeros(uint i)
        {
            uint y;
            if (i == 0) { return 32; }
            uint n = 31;
            y = i << 16; if (y != 0) { n = n - 16; i = y; }
            y = i << 8; if (y != 0) { n = n - 8; i = y; }
            y = i << 4; if (y != 0) { n = n - 4; i = y; }
            y = i << 2; if (y != 0) { n = n - 2; i = y; }
            return n - ((i << 1) >> 31);
        }
    }
}