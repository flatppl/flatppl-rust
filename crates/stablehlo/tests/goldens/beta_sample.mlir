module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %0, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %0, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128xui32>
    %18 = stablehlo.convert %17 : (tensor<128xui32>) -> tensor<128xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128xf32>
    %25 = chlo.erf_inv %24 : tensor<128xf32> -> tensor<128xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128xf32>
    %28 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %29 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %30 = stablehlo.multiply %27, %28 : tensor<128xf32>
    %31 = stablehlo.add %30, %29 : tensor<128xf32>
    %32, %33 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %34 = stablehlo.constant dense<9> : tensor<128xui32>
    %35 = stablehlo.shift_right_logical %33, %34 : tensor<128xui32>
    %36 = stablehlo.convert %35 : (tensor<128xui32>) -> tensor<128xf32>
    %37 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %38 = stablehlo.multiply %36, %37 : tensor<128xf32>
    %39 = stablehlo.subtract %4, %3 : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %41 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %42 = stablehlo.multiply %38, %40 : tensor<128xf32>
    %43 = stablehlo.add %42, %41 : tensor<128xf32>
    %44 = stablehlo.constant dense<0> : tensor<i32>
    %45 = stablehlo.constant dense<false> : tensor<i1>
    %46 = stablehlo.constant dense<0.0> : tensor<f32>
    %50:3 = stablehlo.while(%47 = %44, %48 = %45, %49 = %46) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %51 = stablehlo.constant dense<128> : tensor<i32>
      %52 = stablehlo.compare LT, %47, %51, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %53 = stablehlo.not %48 : tensor<i1>
      %54 = stablehlo.and %53, %52 : tensor<i1>
      stablehlo.return %54 : tensor<i1>
    } do {
      %55 = stablehlo.dynamic_slice %31, %47, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %56 = stablehlo.reshape %55 : (tensor<1xf32>) -> tensor<f32>
      %57 = stablehlo.dynamic_slice %43, %47, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
      %59 = stablehlo.multiply %13, %56 : tensor<f32>
      %60 = stablehlo.add %4, %59 : tensor<f32>
      %61 = stablehlo.multiply %60, %60 : tensor<f32>
      %62 = stablehlo.multiply %61, %60 : tensor<f32>
      %63 = stablehlo.multiply %9, %62 : tensor<f32>
      %64 = stablehlo.constant dense<0.5> : tensor<f32>
      %65 = stablehlo.multiply %56, %56 : tensor<f32>
      %66 = stablehlo.multiply %64, %65 : tensor<f32>
      %67 = stablehlo.multiply %9, %62 : tensor<f32>
      %68 = stablehlo.negate %67 : tensor<f32>
      %69 = stablehlo.log %62 : tensor<f32>
      %70 = stablehlo.multiply %9, %69 : tensor<f32>
      %71 = stablehlo.add %66, %9 : tensor<f32>
      %72 = stablehlo.add %71, %68 : tensor<f32>
      %73 = stablehlo.add %72, %70 : tensor<f32>
      %74 = stablehlo.log %58 : tensor<f32>
      %75 = stablehlo.compare LT, %74, %73 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %76 = stablehlo.compare GT, %62, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %77 = stablehlo.and %75, %76 : tensor<i1>
      %78 = stablehlo.constant dense<1> : tensor<i32>
      %79 = stablehlo.add %47, %78 : tensor<i32>
      stablehlo.return %79, %77, %63 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %80, %81 = stablehlo.rng_bit_generator %32, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %82 = stablehlo.constant dense<9> : tensor<ui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<ui32>
    %84 = stablehlo.convert %83 : (tensor<ui32>) -> tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %86 = stablehlo.multiply %84, %85 : tensor<f32>
    %87 = stablehlo.subtract %4, %3 : tensor<f32>
    %88 = stablehlo.multiply %86, %87 : tensor<f32>
    %89 = stablehlo.add %88, %3 : tensor<f32>
    %90 = stablehlo.divide %4, %0 : tensor<f32>
    %91 = stablehlo.power %89, %90 : tensor<f32>
    %92 = stablehlo.select %5, %91, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %93 = stablehlo.multiply %50#2, %92 : tensor<f32>
    %94 = stablehlo.divide %93, %2 : tensor<f32>
    %95 = stablehlo.constant dense<0.0> : tensor<f32>
    %96 = stablehlo.constant dense<1.0> : tensor<f32>
    %97 = stablehlo.compare LT, %1, %96 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %98 = stablehlo.add %1, %96 : tensor<f32>
    %99 = stablehlo.select %97, %98, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %100 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %101 = stablehlo.subtract %99, %100 : tensor<f32>
    %102 = stablehlo.constant dense<9.0> : tensor<f32>
    %103 = stablehlo.multiply %102, %101 : tensor<f32>
    %104 = stablehlo.sqrt %103 : tensor<f32>
    %105 = stablehlo.divide %96, %104 : tensor<f32>
    %106, %107 = stablehlo.rng_bit_generator %80, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %108 = stablehlo.constant dense<9> : tensor<128xui32>
    %109 = stablehlo.shift_right_logical %107, %108 : tensor<128xui32>
    %110 = stablehlo.convert %109 : (tensor<128xui32>) -> tensor<128xf32>
    %111 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %112 = stablehlo.multiply %110, %111 : tensor<128xf32>
    %113 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %114 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %115 = stablehlo.multiply %112, %113 : tensor<128xf32>
    %116 = stablehlo.subtract %115, %114 : tensor<128xf32>
    %117 = chlo.erf_inv %116 : tensor<128xf32> -> tensor<128xf32>
    %118 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %119 = stablehlo.multiply %117, %118 : tensor<128xf32>
    %120 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %121 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %122 = stablehlo.multiply %119, %120 : tensor<128xf32>
    %123 = stablehlo.add %122, %121 : tensor<128xf32>
    %124, %125 = stablehlo.rng_bit_generator %106, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %126 = stablehlo.constant dense<9> : tensor<128xui32>
    %127 = stablehlo.shift_right_logical %125, %126 : tensor<128xui32>
    %128 = stablehlo.convert %127 : (tensor<128xui32>) -> tensor<128xf32>
    %129 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %130 = stablehlo.multiply %128, %129 : tensor<128xf32>
    %131 = stablehlo.subtract %96, %95 : tensor<f32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %133 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %134 = stablehlo.multiply %130, %132 : tensor<128xf32>
    %135 = stablehlo.add %134, %133 : tensor<128xf32>
    %136 = stablehlo.constant dense<0> : tensor<i32>
    %137 = stablehlo.constant dense<false> : tensor<i1>
    %138 = stablehlo.constant dense<0.0> : tensor<f32>
    %142:3 = stablehlo.while(%139 = %136, %140 = %137, %141 = %138) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %143 = stablehlo.constant dense<128> : tensor<i32>
      %144 = stablehlo.compare LT, %139, %143, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %145 = stablehlo.not %140 : tensor<i1>
      %146 = stablehlo.and %145, %144 : tensor<i1>
      stablehlo.return %146 : tensor<i1>
    } do {
      %147 = stablehlo.dynamic_slice %123, %139, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %148 = stablehlo.reshape %147 : (tensor<1xf32>) -> tensor<f32>
      %149 = stablehlo.dynamic_slice %135, %139, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %150 = stablehlo.reshape %149 : (tensor<1xf32>) -> tensor<f32>
      %151 = stablehlo.multiply %105, %148 : tensor<f32>
      %152 = stablehlo.add %96, %151 : tensor<f32>
      %153 = stablehlo.multiply %152, %152 : tensor<f32>
      %154 = stablehlo.multiply %153, %152 : tensor<f32>
      %155 = stablehlo.multiply %101, %154 : tensor<f32>
      %156 = stablehlo.constant dense<0.5> : tensor<f32>
      %157 = stablehlo.multiply %148, %148 : tensor<f32>
      %158 = stablehlo.multiply %156, %157 : tensor<f32>
      %159 = stablehlo.multiply %101, %154 : tensor<f32>
      %160 = stablehlo.negate %159 : tensor<f32>
      %161 = stablehlo.log %154 : tensor<f32>
      %162 = stablehlo.multiply %101, %161 : tensor<f32>
      %163 = stablehlo.add %158, %101 : tensor<f32>
      %164 = stablehlo.add %163, %160 : tensor<f32>
      %165 = stablehlo.add %164, %162 : tensor<f32>
      %166 = stablehlo.log %150 : tensor<f32>
      %167 = stablehlo.compare LT, %166, %165 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %168 = stablehlo.compare GT, %154, %95 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %169 = stablehlo.and %167, %168 : tensor<i1>
      %170 = stablehlo.constant dense<1> : tensor<i32>
      %171 = stablehlo.add %139, %170 : tensor<i32>
      stablehlo.return %171, %169, %155 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %172, %173 = stablehlo.rng_bit_generator %124, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %174 = stablehlo.constant dense<9> : tensor<ui32>
    %175 = stablehlo.shift_right_logical %173, %174 : tensor<ui32>
    %176 = stablehlo.convert %175 : (tensor<ui32>) -> tensor<f32>
    %177 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %178 = stablehlo.multiply %176, %177 : tensor<f32>
    %179 = stablehlo.subtract %96, %95 : tensor<f32>
    %180 = stablehlo.multiply %178, %179 : tensor<f32>
    %181 = stablehlo.add %180, %95 : tensor<f32>
    %182 = stablehlo.divide %96, %1 : tensor<f32>
    %183 = stablehlo.power %181, %182 : tensor<f32>
    %184 = stablehlo.select %97, %183, %96 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %185 = stablehlo.multiply %142#2, %184 : tensor<f32>
    %186 = stablehlo.divide %185, %2 : tensor<f32>
    %187 = stablehlo.add %94, %186 : tensor<f32>
    %188 = stablehlo.divide %94, %187 : tensor<f32>
    return %188, %172 : tensor<f32>, tensor<2xui64>
  }
}
