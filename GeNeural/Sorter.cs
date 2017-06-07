using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural
{
    public static class Sorter
    {
        private static void Swap<T>(T[] array, int i1, int i2)
        {
            T item = array[i1];
            array[i1] = array[i2];
            array[i2] = item;
        }
        public static void QuickSort<T,D>(T[] array1, D[] array2, int minIndex, int maxIndex) where T :IComparable<T>
        {
            int i = minIndex, j = maxIndex;
            T pivot = array1[(minIndex + maxIndex) / 2];

            while (i <= j)
            {
                while (array1[i].CompareTo(pivot) < 0)
                {
                    i++;
                }

                while (array1[j].CompareTo(pivot) > 0)
                {
                    j--;
                }

                if (i <= j)
                {
                    Swap(array1, i, j);
                    Swap(array2, i, j);
                    i++;
                    j--;
                }
            }
            if (minIndex < j)
            {
                QuickSort(array1, array2, minIndex, j);
            }

            if (i < maxIndex)
            {
                QuickSort(array1,array2, i, maxIndex);
            }
        }

            public static void QuickSort<T,D>(T[] array, D[] array2) where T : IComparable<T>
        {
            QuickSort(array, array2, 0, array.Length - 1);
        }
        public static void SelectionSort<T,D>(T[] array, D[] array2) where T : IComparable<T>
        {
            SelectionSort(array, array2, 0 , array.Length - 1);
        }
        public static void SelectionSort<T, D>(T[] array, D[] array2, int minIndex, int maxIndex) where T : IComparable<T>
        {
            // TODO: Better exception handling
            if (array.Length != array2.Length) { throw new Exception(); }
            for(int i = minIndex; i <= maxIndex;i++)
            {
                int minValueIndex = i;
                for(int j = i + 1; j <= maxIndex; j++)
                {
                    if(array[minValueIndex].CompareTo(array[j]) > 0)
                    {
                        minValueIndex = j;
                    }
                }
                Swap(array, i, minValueIndex);
                Swap(array2, i , minValueIndex);
            }
        }
    }
}
