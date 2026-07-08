module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<2.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.divide %3, %2 : tensor<f32>
    %5 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.compare LT, %4, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = stablehlo.add %4, %6 : tensor<f32>
    %9 = stablehlo.select %7, %8, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %11 = stablehlo.subtract %9, %10 : tensor<f32>
    %12 = stablehlo.constant dense<9.0> : tensor<f32>
    %13 = stablehlo.multiply %12, %11 : tensor<f32>
    %14 = stablehlo.sqrt %13 : tensor<f32>
    %15 = stablehlo.divide %6, %14 : tensor<f32>
    %16, %17 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %18 = stablehlo.constant dense<9> : tensor<128xui32>
    %19 = stablehlo.shift_right_logical %17, %18 : tensor<128xui32>
    %20 = stablehlo.convert %19 : (tensor<128xui32>) -> tensor<128xf32>
    %21 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128xf32>
    %23 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %24 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %25 = stablehlo.multiply %22, %23 : tensor<128xf32>
    %26 = stablehlo.subtract %25, %24 : tensor<128xf32>
    %27 = chlo.erf_inv %26 : tensor<128xf32> -> tensor<128xf32>
    %28 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128xf32>
    %30 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %31 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %32 = stablehlo.multiply %29, %30 : tensor<128xf32>
    %33 = stablehlo.add %32, %31 : tensor<128xf32>
    %34, %35 = stablehlo.rng_bit_generator %16, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %36 = stablehlo.constant dense<9> : tensor<128xui32>
    %37 = stablehlo.shift_right_logical %35, %36 : tensor<128xui32>
    %38 = stablehlo.convert %37 : (tensor<128xui32>) -> tensor<128xf32>
    %39 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<128xf32>
    %41 = stablehlo.subtract %6, %5 : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %43 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %44 = stablehlo.multiply %40, %42 : tensor<128xf32>
    %45 = stablehlo.add %44, %43 : tensor<128xf32>
    %46 = stablehlo.constant dense<0> : tensor<i32>
    %47 = stablehlo.constant dense<false> : tensor<i1>
    %48 = stablehlo.constant dense<0.0> : tensor<f32>
    %52:3 = stablehlo.while(%49 = %46, %50 = %47, %51 = %48) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %53 = stablehlo.constant dense<128> : tensor<i32>
      %54 = stablehlo.compare LT, %49, %53, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %55 = stablehlo.not %50 : tensor<i1>
      %56 = stablehlo.and %55, %54 : tensor<i1>
      stablehlo.return %56 : tensor<i1>
    } do {
      %57 = stablehlo.dynamic_slice %33, %49, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
      %59 = stablehlo.dynamic_slice %45, %49, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %60 = stablehlo.reshape %59 : (tensor<1xf32>) -> tensor<f32>
      %61 = stablehlo.multiply %15, %58 : tensor<f32>
      %62 = stablehlo.add %6, %61 : tensor<f32>
      %63 = stablehlo.multiply %62, %62 : tensor<f32>
      %64 = stablehlo.multiply %63, %62 : tensor<f32>
      %65 = stablehlo.multiply %11, %64 : tensor<f32>
      %66 = stablehlo.constant dense<0.5> : tensor<f32>
      %67 = stablehlo.multiply %58, %58 : tensor<f32>
      %68 = stablehlo.multiply %66, %67 : tensor<f32>
      %69 = stablehlo.multiply %11, %64 : tensor<f32>
      %70 = stablehlo.negate %69 : tensor<f32>
      %71 = stablehlo.log %64 : tensor<f32>
      %72 = stablehlo.multiply %11, %71 : tensor<f32>
      %73 = stablehlo.add %68, %11 : tensor<f32>
      %74 = stablehlo.add %73, %70 : tensor<f32>
      %75 = stablehlo.add %74, %72 : tensor<f32>
      %76 = stablehlo.log %60 : tensor<f32>
      %77 = stablehlo.compare LT, %76, %75 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %78 = stablehlo.compare GT, %64, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %79 = stablehlo.and %77, %78 : tensor<i1>
      %80 = stablehlo.constant dense<1> : tensor<i32>
      %81 = stablehlo.add %49, %80 : tensor<i32>
      stablehlo.return %81, %79, %65 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %82, %83 = stablehlo.rng_bit_generator %34, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %84 = stablehlo.constant dense<9> : tensor<ui32>
    %85 = stablehlo.shift_right_logical %83, %84 : tensor<ui32>
    %86 = stablehlo.convert %85 : (tensor<ui32>) -> tensor<f32>
    %87 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %88 = stablehlo.multiply %86, %87 : tensor<f32>
    %89 = stablehlo.subtract %6, %5 : tensor<f32>
    %90 = stablehlo.multiply %88, %89 : tensor<f32>
    %91 = stablehlo.add %90, %5 : tensor<f32>
    %92 = stablehlo.divide %6, %4 : tensor<f32>
    %93 = stablehlo.power %91, %92 : tensor<f32>
    %94 = stablehlo.select %7, %93, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %95 = stablehlo.multiply %52#2, %94 : tensor<f32>
    %96 = stablehlo.divide %95, %3 : tensor<f32>
    %97 = stablehlo.power %96, %4 : tensor<f32>
    %98 = stablehlo.constant dense<0.0> : tensor<f32>
    %99, %100 = stablehlo.rng_bit_generator %82, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %101 = stablehlo.constant dense<9> : tensor<ui32>
    %102 = stablehlo.shift_right_logical %100, %101 : tensor<ui32>
    %103 = stablehlo.convert %102 : (tensor<ui32>) -> tensor<f32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %105 = stablehlo.multiply %103, %104 : tensor<f32>
    %106 = stablehlo.subtract %3, %98 : tensor<f32>
    %107 = stablehlo.multiply %105, %106 : tensor<f32>
    %108 = stablehlo.add %107, %98 : tensor<f32>
    %109 = stablehlo.constant dense<0.5> : tensor<f32>
    %110 = stablehlo.subtract %108, %109 : tensor<f32>
    %111 = stablehlo.compare GE, %110, %98 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %112 = stablehlo.constant dense<1.0> : tensor<f32>
    %113 = stablehlo.constant dense<-1.0> : tensor<f32>
    %114 = stablehlo.select %111, %112, %113 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %115 = stablehlo.multiply %1, %114 : tensor<f32>
    %116 = stablehlo.multiply %115, %97 : tensor<f32>
    %117 = stablehlo.add %0, %116 : tensor<f32>
    return %117, %99 : tensor<f32>, tensor<2xui64>
  }
}
