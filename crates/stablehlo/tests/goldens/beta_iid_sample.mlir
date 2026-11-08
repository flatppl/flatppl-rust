module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %7 = stablehlo.constant dense<"0x5555D53F"> : tensor<f32>
    %11 = stablehlo.constant dense<"0xA532843E"> : tensor<f32>
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
    %24 = stablehlo.constant dense<1.4142135623730951> : tensor<128x4xf32>
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
    %85 = stablehlo.log %39#2 : tensor<4xf32>
    %89 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %90 = stablehlo.add %85, %89 : tensor<4xf32>
    %91 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %93 = stablehlo.subtract %90, %92 : tensor<4xf32>
    %95 = stablehlo.constant dense<"0xABAA2A40"> : tensor<f32>
    %98 = stablehlo.constant dense<"0xEB05513E"> : tensor<f32>
    %99, %100 = stablehlo.rng_bit_generator %78, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %101 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %102 = stablehlo.shift_right_logical %100, %101 : tensor<128x4xui32>
    %103 = stablehlo.convert %102 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<128x4xf32>
    %106 = stablehlo.multiply %105, %19 : tensor<128x4xf32>
    %107 = stablehlo.subtract %106, %20 : tensor<128x4xf32>
    %108 = chlo.erf_inv %107 : tensor<128x4xf32> -> tensor<128x4xf32>
    %109 = stablehlo.multiply %108, %24 : tensor<128x4xf32>
    %110, %111 = stablehlo.rng_bit_generator %99, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %112 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %113 = stablehlo.shift_right_logical %111, %112 : tensor<128x4xui32>
    %114 = stablehlo.convert %113 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %115 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %116 = stablehlo.multiply %114, %115 : tensor<128x4xf32>
    %117 = stablehlo.constant dense<false> : tensor<4xi1>
    %121:3 = stablehlo.while(%118 = %33, %119 = %117, %120 = %35) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %122 = stablehlo.constant dense<128> : tensor<i32>
      %123 = stablehlo.compare LT, %118, %122, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %124 = stablehlo.constant dense<true> : tensor<i1>
      %125 = stablehlo.reduce(%119 init: %124) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %126 = stablehlo.not %125 : tensor<i1>
      %127 = stablehlo.and %123, %126 : tensor<i1>
      stablehlo.return %127 : tensor<i1>
    } do {
      %128 = stablehlo.constant dense<0> : tensor<i32>
      %129 = stablehlo.dynamic_slice %109, %118, %128, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %130 = stablehlo.reshape %129 : (tensor<1x4xf32>) -> tensor<4xf32>
      %131 = stablehlo.dynamic_slice %116, %118, %128, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %132 = stablehlo.reshape %131 : (tensor<1x4xf32>) -> tensor<4xf32>
      %133 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %134 = stablehlo.multiply %133, %130 : tensor<4xf32>
      %135 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %136 = stablehlo.add %135, %134 : tensor<4xf32>
      %137 = stablehlo.multiply %136, %136 : tensor<4xf32>
      %138 = stablehlo.multiply %137, %136 : tensor<4xf32>
      %139 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %140 = stablehlo.multiply %139, %138 : tensor<4xf32>
      %141 = stablehlo.constant dense<0.5> : tensor<f32>
      %142 = stablehlo.multiply %130, %130 : tensor<4xf32>
      %143 = stablehlo.broadcast_in_dim %141, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %144 = stablehlo.multiply %143, %142 : tensor<4xf32>
      %145 = stablehlo.negate %140 : tensor<4xf32>
      %146 = stablehlo.log %138 : tensor<4xf32>
      %147 = stablehlo.multiply %139, %146 : tensor<4xf32>
      %148 = stablehlo.add %144, %139 : tensor<4xf32>
      %149 = stablehlo.add %148, %145 : tensor<4xf32>
      %150 = stablehlo.add %149, %147 : tensor<4xf32>
      %151 = stablehlo.log %132 : tensor<4xf32>
      %152 = stablehlo.compare LT, %151, %150 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %153 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %154 = stablehlo.compare GT, %138, %153 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %155 = stablehlo.and %152, %154 : tensor<4xi1>
      %156 = stablehlo.select %119, %120, %140 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %157 = stablehlo.or %119, %155 : tensor<4xi1>
      %158 = stablehlo.constant dense<1> : tensor<i32>
      %159 = stablehlo.add %118, %158 : tensor<i32>
      stablehlo.return %159, %157, %156 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %160, %161 = stablehlo.rng_bit_generator %110, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %162 = stablehlo.constant dense<9> : tensor<4xui32>
    %163 = stablehlo.shift_right_logical %161, %162 : tensor<4xui32>
    %164 = stablehlo.convert %163 : (tensor<4xui32>) -> tensor<4xf32>
    %165 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %166 = stablehlo.multiply %164, %165 : tensor<4xf32>
    %167 = stablehlo.log %121#2 : tensor<4xf32>
    %171 = stablehlo.add %167, %89 : tensor<4xf32>
    %172 = stablehlo.subtract %171, %92 : tensor<4xf32>
    %173 = stablehlo.subtract %172, %93 : tensor<4xf32>
    %174 = stablehlo.exponential %173 : tensor<4xf32>
    %175 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %176 = stablehlo.add %175, %174 : tensor<4xf32>
    %177 = stablehlo.divide %175, %176 : tensor<4xf32>
    return %177, %160 : tensor<4xf32>, tensor<2xui64>
  }
}
