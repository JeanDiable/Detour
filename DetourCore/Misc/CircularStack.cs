namespace DetourCore.Misc
{
    public class CircularStack<T>
    {
        private int stack_bottom;
        private int stack_top;
        private int N = 1500;
        private T[] arr;

        private object sync = new object();

        public CircularStack(int N = 1024)
        {
            arr = new T[N];
            this.N = N;
        }

        public void Clear()
        {
            lock (sync)
                stack_bottom = stack_top;
        }

        public bool NotEmpty()
        {
            lock (sync)
                return stack_top != stack_top;
        }

        public void Push(T what)
        {
            lock (sync)
            {
                arr[stack_top] = what;
                stack_top += 1;
                if (stack_top == N) stack_top = 0;
                if (stack_top == stack_bottom)
                { // WRONG
                    stack_bottom += 1;
                    if (stack_bottom == N)
                        stack_bottom = 0;
                }
            }
        }

        public T Peek(int n=1)
        {
            if (stack_bottom == stack_top)
                return default(T);
            int vt=stack_top - n;
            if (vt <= -1) vt += N;
            return arr[vt];
        }

        public bool TryPop(out T what)
        {
            lock (sync)
            {
                if (stack_bottom == stack_top)
                {
                    what = default(T);
                    return false;
                }

                stack_top -= 1;
                if (stack_top == -1) stack_top = N - 1;
                what = arr[stack_top];
                return true;
            }
        }

        public int Size()
        {
            var sz = stack_top - stack_bottom;
            if (sz < 0) sz += N;
            return sz;
        }
    }
}