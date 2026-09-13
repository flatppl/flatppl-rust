module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.compare LT, %0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.select %4, %1, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %6 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %7 = stablehlo.subtract %5, %6 : tensor<f32>
    %8 = stablehlo.constant dense<9.0> : tensor<f32>
    %9 = stablehlo.multiply %8, %7 : tensor<f32>
    %10 = stablehlo.sqrt %9 : tensor<f32>
    %11 = stablehlo.divide %2, %10 : tensor<f32>
    %12, %13 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %14 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<128x4xui32>
    %16 = stablehlo.convert %15 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %17 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x4xf32>
    %19 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %20 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %21 = stablehlo.multiply %18, %19 : tensor<128x4xf32>
    %22 = stablehlo.subtract %21, %20 : tensor<128x4xf32>
    %23 = chlo.erf_inv %22 : tensor<128x4xf32> -> tensor<128x4xf32>
    %24 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<128x4xf32>
    %26, %27 = stablehlo.rng_bit_generator %12, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %28 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %29 = stablehlo.shift_right_logical %27, %28 : tensor<128x4xui32>
    %30 = stablehlo.convert %29 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %31 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<128x4xf32>
    %33 = stablehlo.constant dense<0> : tensor<i32>
    %34 = stablehlo.constant dense<false> : tensor<4xi1>
    %35 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %39:3 = stablehlo.while(%36 = %33, %37 = %34, %38 = %35) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %40 = stablehlo.constant dense<128> : tensor<i32>
      %41 = stablehlo.compare LT, %36, %40, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %42 = stablehlo.constant dense<true> : tensor<i1>
      %43 = stablehlo.reduce(%37 init: %42) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %44 = stablehlo.not %43 : tensor<i1>
      %45 = stablehlo.and %41, %44 : tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %46 = stablehlo.constant dense<0> : tensor<i32>
      %47 = stablehlo.dynamic_slice %25, %36, %46, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %48 = stablehlo.reshape %47 : (tensor<1x4xf32>) -> tensor<4xf32>
      %49 = stablehlo.dynamic_slice %32, %36, %46, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x4xf32>) -> tensor<4xf32>
      %51 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %52 = stablehlo.multiply %51, %48 : tensor<4xf32>
      %53 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %54 = stablehlo.add %53, %52 : tensor<4xf32>
      %55 = stablehlo.multiply %54, %54 : tensor<4xf32>
      %56 = stablehlo.multiply %55, %54 : tensor<4xf32>
      %57 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %58 = stablehlo.multiply %57, %56 : tensor<4xf32>
      %59 = stablehlo.constant dense<0.5> : tensor<f32>
      %60 = stablehlo.multiply %48, %48 : tensor<4xf32>
      %61 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %62 = stablehlo.multiply %61, %60 : tensor<4xf32>
      %63 = stablehlo.negate %58 : tensor<4xf32>
      %64 = stablehlo.log %56 : tensor<4xf32>
      %65 = stablehlo.multiply %57, %64 : tensor<4xf32>
      %66 = stablehlo.add %62, %57 : tensor<4xf32>
      %67 = stablehlo.add %66, %63 : tensor<4xf32>
      %68 = stablehlo.add %67, %65 : tensor<4xf32>
      %69 = stablehlo.log %50 : tensor<4xf32>
      %70 = stablehlo.compare LT, %69, %68 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %71 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %72 = stablehlo.compare GT, %56, %71 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %73 = stablehlo.and %70, %72 : tensor<4xi1>
      %74 = stablehlo.select %37, %38, %58 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %75 = stablehlo.or %37, %73 : tensor<4xi1>
      %76 = stablehlo.constant dense<1> : tensor<i32>
      %77 = stablehlo.add %36, %76 : tensor<i32>
      stablehlo.return %77, %75, %74 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %78, %79 = stablehlo.rng_bit_generator %26, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %80 = stablehlo.constant dense<9> : tensor<4xui32>
    %81 = stablehlo.shift_right_logical %79, %80 : tensor<4xui32>
    %82 = stablehlo.convert %81 : (tensor<4xui32>) -> tensor<4xf32>
    %83 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %84 = stablehlo.multiply %82, %83 : tensor<4xf32>
    %85 = stablehlo.constant dense<0.5> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %87 = stablehlo.power %84, %86 : tensor<4xf32>
    %88 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %89 = stablehlo.select %4, %87, %88 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %90 = stablehlo.multiply %39#2, %89 : tensor<4xf32>
    %91 = stablehlo.divide %90, %88 : tensor<4xf32>
    %92 = stablehlo.compare LT, %1, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %93 = stablehlo.constant dense<4.0> : tensor<f32>
    %94 = stablehlo.select %92, %93, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %95 = stablehlo.subtract %94, %6 : tensor<f32>
    %96 = stablehlo.multiply %8, %95 : tensor<f32>
    %97 = stablehlo.sqrt %96 : tensor<f32>
    %98 = stablehlo.divide %2, %97 : tensor<f32>
    %99, %100 = stablehlo.rng_bit_generator %78, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %101 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %102 = stablehlo.shift_right_logical %100, %101 : tensor<128x4xui32>
    %103 = stablehlo.convert %102 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<128x4xf32>
    %106 = stablehlo.multiply %105, %19 : tensor<128x4xf32>
    %107 = stablehlo.subtract %106, %20 : tensor<128x4xf32>
    %108 = chlo.erf_inv %107 : tensor<128x4xf32> -> tensor<128x4xf32>
    %109 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<128x4xf32>
    %111, %112 = stablehlo.rng_bit_generator %99, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %113 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %114 = stablehlo.shift_right_logical %112, %113 : tensor<128x4xui32>
    %115 = stablehlo.convert %114 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %116 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %117 = stablehlo.multiply %115, %116 : tensor<128x4xf32>
    %118 = stablehlo.constant dense<false> : tensor<4xi1>
    %122:3 = stablehlo.while(%119 = %33, %120 = %118, %121 = %35) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %123 = stablehlo.constant dense<128> : tensor<i32>
      %124 = stablehlo.compare LT, %119, %123, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %125 = stablehlo.constant dense<true> : tensor<i1>
      %126 = stablehlo.reduce(%120 init: %125) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %127 = stablehlo.not %126 : tensor<i1>
      %128 = stablehlo.and %124, %127 : tensor<i1>
      stablehlo.return %128 : tensor<i1>
    } do {
      %129 = stablehlo.constant dense<0> : tensor<i32>
      %130 = stablehlo.dynamic_slice %110, %119, %129, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %131 = stablehlo.reshape %130 : (tensor<1x4xf32>) -> tensor<4xf32>
      %132 = stablehlo.dynamic_slice %117, %119, %129, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %133 = stablehlo.reshape %132 : (tensor<1x4xf32>) -> tensor<4xf32>
      %134 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %135 = stablehlo.multiply %134, %131 : tensor<4xf32>
      %136 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %137 = stablehlo.add %136, %135 : tensor<4xf32>
      %138 = stablehlo.multiply %137, %137 : tensor<4xf32>
      %139 = stablehlo.multiply %138, %137 : tensor<4xf32>
      %140 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %141 = stablehlo.multiply %140, %139 : tensor<4xf32>
      %142 = stablehlo.constant dense<0.5> : tensor<f32>
      %143 = stablehlo.multiply %131, %131 : tensor<4xf32>
      %144 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %145 = stablehlo.multiply %144, %143 : tensor<4xf32>
      %146 = stablehlo.negate %141 : tensor<4xf32>
      %147 = stablehlo.log %139 : tensor<4xf32>
      %148 = stablehlo.multiply %140, %147 : tensor<4xf32>
      %149 = stablehlo.add %145, %140 : tensor<4xf32>
      %150 = stablehlo.add %149, %146 : tensor<4xf32>
      %151 = stablehlo.add %150, %148 : tensor<4xf32>
      %152 = stablehlo.log %133 : tensor<4xf32>
      %153 = stablehlo.compare LT, %152, %151 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %154 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %155 = stablehlo.compare GT, %139, %154 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %156 = stablehlo.and %153, %155 : tensor<4xi1>
      %157 = stablehlo.select %120, %121, %141 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %158 = stablehlo.or %120, %156 : tensor<4xi1>
      %159 = stablehlo.constant dense<1> : tensor<i32>
      %160 = stablehlo.add %119, %159 : tensor<i32>
      stablehlo.return %160, %158, %157 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %161, %162 = stablehlo.rng_bit_generator %111, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %163 = stablehlo.constant dense<9> : tensor<4xui32>
    %164 = stablehlo.shift_right_logical %162, %163 : tensor<4xui32>
    %165 = stablehlo.convert %164 : (tensor<4xui32>) -> tensor<4xf32>
    %166 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %167 = stablehlo.multiply %165, %166 : tensor<4xf32>
    %168 = stablehlo.constant dense<0.3333333432674408> : tensor<f32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %170 = stablehlo.power %167, %169 : tensor<4xf32>
    %171 = stablehlo.select %92, %170, %88 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %172 = stablehlo.multiply %122#2, %171 : tensor<4xf32>
    %173 = stablehlo.divide %172, %88 : tensor<4xf32>
    %174 = stablehlo.add %91, %173 : tensor<4xf32>
    %175 = stablehlo.divide %91, %174 : tensor<4xf32>
    return %175, %161 : tensor<4xf32>, tensor<2xui64>
  }
}
